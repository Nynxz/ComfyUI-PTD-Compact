"""Compact PTD adapter files: int8 factors with per-row scales and/or reduced ranks.

The adapter bank stays compact in host RAM (int8, true ranks). Each adapted layer moves only its
active hop's int8 factors to the GPU and applies the scales there, so a 4-step bank costs ~2.8 GB of
RAM instead of 5.5 (bf16) or 11.5 (fp32), and no compute is spent on padded ranks.

Everything else (base checks, model clone, raw refiner, hop routing, audio clock, sampler) is the
official ``ptd_comfy`` code: ``patch_ptd_model`` runs unchanged with this controller in place of the
stock one. Standard (float) PTD files go through the official loader untouched.

PTD adds its deltas through forward hooks on the linear outputs, while ordinary LoRAs are weight
patches, so the two stack in either order: a LoRA loaded before or after this node is merged into
the base weights as usual and every hop's PTD delta is added on top.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import functools
import importlib
import json
import logging
import threading
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open
from tqdm.auto import tqdm

from . import pdd

COMPACT = "comfyui-native-h3-compact"
_SWAP = threading.RLock()


@dataclass(frozen=True)
class QExpert:
    a: torch.Tensor                  # [r, in]   int8 or float
    b: torch.Tensor                  # [out, r]  int8 or float
    sa: torch.Tensor | None          # [r]   float32 row scales of a
    sb: torch.Tensor | None          # [out] float32 row scales of b
    qkv_index: int | None


def ptd_modules():
    """(checkpoint, patcher) of the ComfyUI-PTD that ComfyUI registered.

    ComfyUI imports a custom-node directory under a path-derived module name, so its PTD Sampler checks
    ``isinstance`` against ``<pack>.ptd_comfy.patcher.PTDController``, not against a pip-installed
    top-level ``ptd_comfy``. The controller must come from the registered copy.
    """
    try:
        import nodes
        sampler = nodes.NODE_CLASS_MAPPINGS.get("PTDH3Sampler")
    except ImportError:  # outside ComfyUI (tests)
        sampler = None
    if sampler is not None:
        package = sampler.__module__.rsplit(".", 1)[0]
        return importlib.import_module(package + ".checkpoint"), importlib.import_module(package + ".patcher")
    try:
        from ptd_comfy import checkpoint, patcher
    except ImportError as e:
        raise RuntimeError(
            "PTD Compact needs the official ComfyUI-PTD node pack next to it. Run `git clone --branch "
            "release/comfyui-inference https://github.com/vita-epfl/PTD-inference ComfyUI-PTD` inside "
            "ComfyUI/custom_nodes and restart ComfyUI (no pip install needed).") from e
    return checkpoint, patcher


def read(path):
    with safe_open(str(path), framework="pt", device="cpu") as f:
        meta = dict(f.metadata() or {})
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    return tensors, meta


def relaxed_pairs(tensors, recipe, *, expected_blocks=50, expected_refiner_blocks=2, **_):
    """The official topology check without the fixed-rank/float requirements."""
    checkpoint, _ = ptd_modules()
    RECIPE_HOPS, AdapterPair = checkpoint.RECIPE_HOPS, checkpoint.AdapterPair
    _expected_modules, map_adapter_key = checkpoint._expected_modules, checkpoint.map_adapter_key

    hops = RECIPE_HOPS[recipe]
    pairs, seen = [], set()
    for key in tensors:
        if ".lora_A." not in key or key.endswith("_scale"):
            continue
        target, hop, _ = map_adapter_key(key)
        b_key = key.replace(".lora_A.", ".lora_B.")
        a, b = tensors[key], tensors[b_key]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1] or hop >= hops:
            raise ValueError(f"Invalid PTD adapter pair: {key}")
        seen.add((key.rsplit(".lora_A.", 1)[0], hop))
        pairs.append(AdapterPair(target, hop, key, b_key))
    expected = {(m, h) for m in _expected_modules(expected_blocks, expected_refiner_blocks) for h in range(hops)}
    if seen != expected:
        raise ValueError(f"Incomplete or unexpected PTD topology: {len(expected - seen)} missing, "
                         f"{len(seen - expected)} unexpected pairs")
    return tuple(sorted(pairs, key=lambda p: (p.a_key, p.hop)))


def _controller_class(patcher):
    class CompactPTDController(patcher.PTDController):
        """``PTDController`` whose experts may be int8 with row scales and any rank."""

        def __init__(self, diffusion_model, tensors, pairs, recipe, hops):
            self.recipe, self.hops = recipe, int(hops)
            self.diffusion_model = diffusion_model
            self._lock = patcher._model_lock(diffusion_model)
            self._selection = ContextVar(f"ptd_selection_{id(self)}", default=None)
            self.interrupt_check = patcher._check_interrupt
            bank = defaultdict(lambda: defaultdict(list))
            for pair in pairs:
                target = pair.target
                module = diffusion_model.get_submodule(target.module)
                a, b = tensors[pair.a_key], tensors[pair.b_key]
                sa, sb = tensors.get(pair.a_key + "_scale"), tensors.get(pair.b_key + "_scale")
                if a.shape[1] != module.in_features or \
                        b.shape[0] * (3 if target.qkv_index is not None else 1) != module.out_features:
                    raise ValueError(f"PTD adapter does not fit {target.module}")
                if target.swap_halves:  # Diffusers stores [up, gate]; native H3 consumes [gate, up]
                    b = torch.cat(b.chunk(2, dim=0)[::-1], dim=0)
                    sb = None if sb is None else torch.cat(sb.chunk(2, dim=0)[::-1], dim=0)
                bank[target.module][pair.hop].append(QExpert(
                    a.contiguous(), b.contiguous(), None if sa is None else sa.reshape(-1).float(),
                    None if sb is None else sb.reshape(-1).float(), target.qkv_index))
            self._bank = {name: {hop: tuple(e) for hop, e in by_hop.items()} for name, by_hop in bank.items()}
            for name, by_hop in self._bank.items():
                if set(by_hop) != set(range(self.hops)):
                    raise ValueError(f"PTD adapter is missing hops for {name}")

        fuse_blocks = False  # run the native fused block/attention code instead of PTD's step-by-step math
        allocation_compiler = True  # keep ComfyUI's allocation compiler on; off for a run it failed in (_with_progress)

        @contextmanager
        def forward_hooks(self, selection):
            with ExitStack() as stack:
                if not self.allocation_compiler:
                    stack.enter_context(_without_comfy_compiler())
                # The official forward_hooks reads _scope_forward only while it sets up; swap it for that.
                with _SWAP:
                    scope = patcher._scope_forward
                    if self.fuse_blocks:
                        patcher._scope_forward = lambda *args, **kwargs: None
                    try:
                        stack.enter_context(super().forward_hooks(selection))
                    finally:
                        patcher._scope_forward = scope
                yield

        def _linear_hook(self, frame, experts):
            # q/k/v share one input: their A factors run as a single matmul. Each expert's deltas are
            # added into the fresh linear output in place, and the int8 row scales are folded into the
            # small factor matrices instead of scaling the large (tokens x out) results.
            qkv = sorted((e for e in experts if e.qkv_index is not None), key=lambda e: e.qkv_index)
            plain = [e for e in experts if e.qkv_index is None]
            ranks = [e.a.shape[0] for e in qkv]

            def factors(e, dev, dt):
                a = e.a.to(dev, non_blocking=True).to(dt)
                b = e.b.to(dev, non_blocking=True).to(dt)
                if e.sa is not None:
                    a = a * e.sa.to(dev, dt)[:, None]
                if e.sb is not None:
                    b = b * e.sb.to(dev, dt)[:, None]
                return a, b

            def hook(module, args, kwargs, output):
                if patcher._FORWARD.get() is not frame:
                    return output
                self.interrupt_check()
                x = args[0] if args else kwargs.get("input")
                dev, dt = x.device, output.dtype   # the layer's compute dtype, as in the stock hook
                x = x.to(dt)
                for e in plain:
                    a, b = factors(e, dev, dt)
                    output.add_(F.linear(F.linear(x, a), b))
                if qkv:
                    pairs = [factors(e, dev, dt) for e in qkv]
                    hs = F.linear(x, torch.cat([a for a, _ in pairs])).split(ranks, dim=-1)
                    w = output.shape[-1] // 3
                    for e, (_, b), h in zip(qkv, pairs, hs):
                        output[..., e.qkv_index * w:(e.qkv_index + 1) * w].add_(F.linear(h, b))
                return output
            return hook

    return CompactPTDController


@contextmanager
def _without_comfy_compiler():
    """Switch off ComfyUI's allocation compiler (and its CUDA graphs) for one PTD model call.

    Used only for a run in which the compiler failed (``allocation_compiler``), see _with_progress.

    With DynamicVRAM the H3 forward records each block's GPU allocations into a malloc graph and replays
    it. PTD's scoped block math and adapter hooks allocate extra tensors inside every block, which that
    recording does not expect ("aimdo memory compile error"). Weight streaming stays on; only the
    allocation compiler is skipped while a PTD hop runs, and it is restored afterwards.
    """
    try:
        from comfy.cli_args import args
    except ImportError:
        yield
        return
    saved = args.disable_comfy_compiler, args.disable_cuda_graphs
    args.disable_comfy_compiler = args.disable_cuda_graphs = True
    try:
        yield
    finally:
        args.disable_comfy_compiler, args.disable_cuda_graphs = saved


@contextmanager
def _accept_curve_base(model):
    """Let the official checks accept a pruned (AdaLN-curve) base for the duration of patching.

    Comfy-Org's pruned H3 keeps every adapted linear and both output heads byte-identical to the full
    model and replaces only time_embedder + adaln_proj with an 8-wide curve table. At PTD's hop times that
    curve reproduces the original AdaLN modulation to 0.03% (the original's own bf16 path is off by 0.2%).
    PTD's time-label hooks already skip models without a time embedder, so the forward runs as on the full
    base. The flag is restored right after the checks; the native forward still uses the curves.
    """
    native = model.get_model_object("diffusion_model")
    curve = bool(getattr(native, "use_adaln_curves", False))
    if curve:
        logging.info("PTD Compact: pruned (AdaLN-curve) MiniMax-H3 base; adapted layers are identical to the full model")
        native.use_adaln_curves = False
    try:
        yield
    finally:
        if curve:
            native.use_adaln_curves = True


def _with_progress(sample):
    """``sample_ptd`` plus the console progress bar ComfyUI's own samplers print.

    The official sampler drops ``disable`` and reports hops only through ``callback`` (the UI bar), so the
    console showed nothing per step. Ticking a tqdm bar from that callback gives the usual
    ``6/6 [02:30<00:00, 25.1s/it]`` line, timed per hop.
    """
    @functools.wraps(sample)
    def run(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
        bar = tqdm(total=len(sigmas) - 1, disable=disable)

        done = [0]

        def step(record):
            done[0] += 1
            bar.update(1)
            _between_hops(model, record.get("i"))
            if callback is not None:
                callback(record)

        controller = kwargs.get("controller")
        # Whatever ran before the first hop (VAE decode, text encoders, or a SelfLift transition inside the same
        # node) may have left PyTorch's cache full, which the allocation compiler cannot get when it starts
        # recording, and the base model's watermark lowered, which would stream the first hop from RAM.
        _between_hops(model, "start")
        start = time.perf_counter()
        try:
            try:
                out = sample(model, x, sigmas, extra_args=extra_args, callback=step, disable=disable, **kwargs)
            except RuntimeError as e:
                if "aimdo memory compile error" not in str(e) or not getattr(controller, "allocation_compiler", False):
                    raise
                # A known, intermittent ComfyUI/aimdo failure when VRAM is nearly full (ComfyUI #16342, also with
                # the stock H3 sampler). Drop the half-recorded graph and sample this run again without the
                # compiler; the hops are deterministic, so the result is the same. The next run tries it again.
                logging.warning("PTD: ComfyUI's allocation compiler failed at hop %d (%s); sampling this run "
                                "without it", done[0], e, exc_info=True)
                import comfy.model_prefetch
                comfy.model_prefetch.cleanup_prefetch_queues()
                controller.allocation_compiler = False
                bar.reset()
                done[0] = 0
                start = time.perf_counter()
                try:
                    out = sample(model, x, sigmas, extra_args=extra_args, callback=step, disable=disable, **kwargs)
                finally:
                    controller.allocation_compiler = True
        finally:
            bar.close()
        hops, seconds = len(sigmas) - 1, time.perf_counter() - start
        logging.info("PTD: %d hops in %.1f s (%.2f s/hop)", hops, seconds, seconds / hops)
        return out

    run.ptd_progress = True
    return run


def _between_hops(model, hop):
    """Hand PyTorch's cached activation memory back after a hop, then log where the GPU memory is.

    A hop's activations (~5 GB for a 0.75 MP, 5 s clip) are freed when it ends, but PyTorch keeps the blocks
    cached. When a hop needs that memory, aimdo lowers the base model's VBAR watermark and evicts the weights
    above it, and ComfyUI only resets the watermark once the whole sampler node finishes, so residency could
    only fall from hop to hop. Release the cache and raise the watermark again, as ComfyUI's own LLM decode
    does after its prefill, so evicted weights fault back in during the next hop while there is room.
    """
    try:
        if not torch.cuda.is_available():
            return
        before = torch.cuda.memory_reserved() / 2**30
        torch.cuda.empty_cache()
        patcher = getattr(getattr(model, "inner_model", None), "model_patcher", None)
        vbar = patcher._vbar_get() if hasattr(patcher, "_vbar_get") else None
        if vbar is not None:
            import comfy.model_management
            comfy.model_management.reset_cast_buffers()
            vbar.set_watermark(vbar.max_size)
        resident = patcher.loaded_size() / 2**30 if patcher is not None else float("nan")
        size = patcher.model_size() / 2**30 if patcher is not None else float("nan")
        stats = torch.cuda.memory_stats()
        free, total = torch.cuda.mem_get_info()
        logging.info("PTD hop %s VRAM: base model resident %.1f/%.1f GB; torch allocated %.1f GB, cache %.1f -> %.1f GB "
                     "(peak allocated %.1f GB); card %.1f/%.1f GB used", hop, resident, size,
                     stats.get("allocated_bytes.all.current", 0) / 2**30, before,
                     stats.get("reserved_bytes.all.current", 0) / 2**30,
                     stats.get("allocated_bytes.all.peak", 0) / 2**30, (total - free) / 2**30, total / 2**30)
    except Exception as e:  # memory housekeeping and diagnostics only
        logging.debug("PTD VRAM release failed: %s", e)


def _residual_in_place(state, update, gate, segments):
    """PTD's reference residual (official ``_residual``), added into the stream instead of a new tensor.

    The official version writes each block's output into ``torch.empty_like(state)``. With ComfyUI's
    allocation compiler on, that output is allocated inside one block's scope, outlives it, and is freed
    during the next block, which aborts the recording with ``aimdo memory compile error``. The native
    block updates the stream in place. Here the product is still rounded to bf16 on its own (no addcmul,
    as PTD was trained), and ``add_`` rounds the sum once, exactly as ``state + product`` does.
    """
    for start, stop, row in segments:
        state[start:stop].add_(update[start:stop] * gate[row].to(state.dtype))
    return state


def _install_residual(patcher):
    """Point the registered pack's reference block maths at the in-place residual."""
    if patcher._residual is not _residual_in_place:
        patcher._residual = _residual_in_place


def _sampling(patcher):
    return importlib.import_module(patcher.__name__.rsplit(".", 1)[0] + ".sampling")


_pdd_sample = _with_progress(pdd.sample)


def make_sampler(model):
    """SAMPLER for a PTD MODEL: the checkpoint's own hop grid for Ref2VA files, the official one otherwise."""
    from comfy.samplers import KSAMPLER
    _, patcher = ptd_modules()
    controller = patcher.get_ptd_controller(model)
    if hasattr(controller, "sigmas"):
        return KSAMPLER(_pdd_sample, extra_options={"controller": controller, "sampling": _sampling(patcher)})
    _install_progress(patcher)
    return _sampling(patcher).make_sampler(controller, controller.recipe)


def sigmas(model):
    _, patcher = ptd_modules()
    controller = patcher.get_ptd_controller(model)
    if hasattr(controller, "sigmas"):
        return controller.sigmas.float()
    return _sampling(patcher).ptd_sigmas(controller.recipe)


def _install_official_nodes(patcher):
    """Point the official PTD Sampler / Sigmas at ``make_sampler`` / ``sigmas``.

    They only know the EPFL recipes and fail on a Ref2VA MODEL (``unknown recipe 'pdd_4hop'``). For EPFL files
    ``make_sampler`` and ``sigmas`` return exactly what the official nodes did, so existing workflows keep working.
    """
    official = importlib.import_module(patcher.__name__.rsplit(".", 1)[0] + ".nodes")
    if not getattr(official.PTDH3Sigmas, "ptd_compact", False):
        official.PTDH3Sampler.get_sampler = lambda self, model: (make_sampler(model),)
        official.PTDH3Sigmas.get_sigmas = lambda self, model: (sigmas(model),)
        official.PTDH3Sampler.ptd_compact = official.PTDH3Sigmas.ptd_compact = True


def _install_progress(patcher):
    """Wrap the registered pack's ``sample_ptd``; ``make_sampler`` looks it up when PTD Sampler runs."""
    sampling = _sampling(patcher)
    if not getattr(sampling.sample_ptd, "ptd_progress", False):
        sampling.sample_ptd = _with_progress(sampling.sample_ptd)


# Block arithmetic per hop. exact: PTD's step-by-step bf16 maths, which the experts were trained against.
# fast: ComfyUI's native fused blocks (in-place modulation, fused RMSNorm + RoPE), ~8% faster per block.
# Both are bf16 rounding of the same maths and lie equally close to fp32. fc2 always runs eager, because
# its adapter needs fc2's input (a fused SwiGLU + fc2 kernel is no faster once that input is recomputed).
KERNELS = {"exact (PTD reference)": False, "fast (native kernels)": True}


def patch_compact(model, path, kernels="exact (PTD reference)"):
    """Return a MODEL clone carrying the adapters in ``path`` (compact or standard).

    ``kernels`` picks PTD's reference block arithmetic or ComfyUI's fused kernels (see KERNELS).
    """
    patched = _patch(model, path)
    patched.model_options["ptd_native_controller"].fuse_blocks = KERNELS[kernels]
    return patched


def _patch(model, path):
    checkpoint, patcher = ptd_modules()
    _install_progress(patcher)
    _install_residual(patcher)
    _install_official_nodes(patcher)
    tensors, meta = read(path)
    if pdd.is_pdd(meta):  # taobao-content-ai Ref2VA files (bf16, or this pack's int8 version)
        for key, value in tensors.items():
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"Non-finite values in {key}")
        if getattr(model.get_model_object("diffusion_model"), "use_adaln_curves", False):
            logging.info("PTD Compact: pruned (AdaLN-curve) MiniMax-H3 base; adapted layers are identical to the full model")
        return pdd.patch(model, tensors, meta, patcher, _sampling(patcher), _controller_class(patcher), QExpert)
    if meta.get("ptd_format") == checkpoint.FORMAT:
        tensors, meta = checkpoint.load_adapter_file(path)
        with _SWAP, _accept_curve_base(model):
            stock_cls = patcher.PTDController
            patcher.PTDController = _controller_class(patcher)
            try:
                return patcher.patch_ptd_model(model, tensors, meta)
            finally:
                patcher.PTDController = stock_cls
    if meta.get("ptd_format") != COMPACT:
        raise ValueError(f"Not a PTD adapter file (ptd_format={meta.get('ptd_format')!r})")
    compact = json.loads(meta.get("ptd_compact", "{}"))
    if compact.get("schema") != 1:
        raise ValueError(f"Unsupported compact PTD schema {compact.get('schema')!r}")
    for key, value in tensors.items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"Non-finite values in {key}")
    meta = dict(meta, ptd_format=checkpoint.FORMAT)
    # patch_ptd_model looks both names up at call time; swap them for this call only.
    with _SWAP, _accept_curve_base(model):
        stock_cls, stock_validate = patcher.PTDController, checkpoint.validate_adapter_tensors
        patcher.PTDController, checkpoint.validate_adapter_tensors = _controller_class(patcher), relaxed_pairs
        try:
            return patcher.patch_ptd_model(model, tensors, meta)
        finally:
            patcher.PTDController, checkpoint.validate_adapter_tensors = stock_cls, stock_validate

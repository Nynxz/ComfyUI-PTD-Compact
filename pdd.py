"""PTD checkpoints trained on PDD output heads: taobao-content-ai/MiniMax-H3-PTD-LoRAs (Ref2VA, 4 / 8 hops).

These differ from the EPFL release in three ways, all handled here:

* the adapter factors are stacked over hops (``lora_down [H, r, in]``, ``lora_up [H, out, r]``), bf16, or
  int8 with per-row scales in the compact version this pack writes (``to_int8``);
* the hop times come from the checkpoint's own 32-interval PDD grid, not an upstream recipe, so 8 hops work;
* the output heads are a bank of ``pdd_num_steps`` per-interval heads, fused per hop with the block's
  normalised step sizes and swapped in for the native heads by a forward hook.

The schedule, head fusion and audio correction follow ComfyUI-MiniMaxH3-PTD (taobao-content-ai, Apache-2.0);
see NOTICE. The adapter maths, hook machinery and block arithmetic are this pack's ``CompactPTDController``
over the official ComfyUI-PTD code, so pruned bases, int8, the ``fast`` kernels and DynamicVRAM work as for
the EPFL files.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
import copy
from itertools import pairwise
import json
import re

import torch
import torch.nn.functional as F

PDD_NUM_STEPS = 32
VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0
T_MAX = 0.999  # upper endpoint of ComfyUI-PTD's shared clock (ptd_comfy.sampling.T_MAX)

PARTS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")
_QKV = {"attn.to_q": 0, "attn.to_k": 1, "attn.to_v": 2}
_DENSE = {"attn.to_out.0": "attn.out_proj", "ff.net.0.proj": "mlp.fc1", "ff.net.2": "mlp.fc2"}
HEADS = {"proj_out": "final_layer.video_out", "audio_proj_out": "final_layer.audio_out"}
_MODULE = re.compile(r"^(transformer_blocks|token_refiner\.refiner_blocks)\.(\d+)\.(.+)$")


def is_pdd(meta):
    return "ptd_num_hops" in meta


# ----------------------------------------------------------------------------------------- schedule

def _shifted(shift, sigma):
    return shift * sigma / (1 + (shift - 1) * sigma)


def _time_grid(shift, steps):
    """Ascending ``0 = t_0 < ... < t_N = 1`` of one MiniMax-H3 schedule."""
    return 1.0 - _shifted(shift, torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64))


def hop_sigmas(hops, steps=PDD_NUM_STEPS, video_shift=VIDEO_SHIFT):
    """One knot per hop plus 0: the video block boundaries of the trained grid, on the T_MAX clock."""
    size = steps // hops
    noise = _shifted(float(video_shift), torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64))
    return torch.tensor([T_MAX * float(noise[h * size]) for h in range(hops)] + [0.0], dtype=torch.float64)


def hop_plans(hops, steps=PDD_NUM_STEPS, video_shift=VIDEO_SHIFT, audio_shift=AUDIO_SHIFT):
    """Per hop, (video, audio) weights over the per-interval heads: the block's normalised step sizes."""
    size = steps // hops
    plans = []
    for shift in (video_shift, audio_shift):
        sizes = _time_grid(float(shift), steps).diff()
        rows = []
        for h in range(hops):
            plan = torch.zeros(steps, dtype=torch.float64)
            block = sizes[h * size:(h + 1) * size]
            plan[h * size:(h + 1) * size] = block / block.sum()
            rows.append(plan.float())
        plans.append(rows)
    return list(zip(*plans))


def audio_corrections(sigmas, audio_sigma, audio_sigma_prime):
    """Per hop, chord / tangent of the audio clock.

    ComfyUI-PTD scales the audio head by the clock's tangent at the hop start; these checkpoints' audio head
    is PDD's block-mean velocity over the hop's whole audio interval, i.e. the chord.
    """
    grid = [float(s) for s in sigmas]
    return tuple((audio_sigma(a) - audio_sigma(b)) / (a - b) / audio_sigma_prime(a) for a, b in pairwise(grid))


# ------------------------------------------------------------------------------------------ file

def _target(module):
    """Diffusers module name -> (native module, qkv index)."""
    m = _MODULE.fullmatch(module)
    if m is None:
        raise ValueError(f"Unknown PTD adapter module: {module}")
    base = ("blocks." if m[1] == "transformer_blocks" else "token_refiner.blocks.") + m[2]
    if m[3] in _QKV:
        return f"{base}.attn.qkv_proj", _QKV[m[3]]
    if m[3] not in _DENSE:
        raise ValueError(f"Unknown PTD adapter projection: {module}")
    return f"{base}.{_DENSE[m[3]]}", None


def load(tensors, meta, expert_cls):
    """Per-hop experts and the PDD head bank from a (bf16 or compact int8) stacked-hop file."""
    hops = int(meta["ptd_num_hops"])
    steps = int(meta.get("pdd_num_steps", PDD_NUM_STEPS))
    rank = int(meta.get("lora_rank", 128))
    scale = float(meta.get("lora_alpha", rank)) / rank
    if hops < 1 or steps % hops:
        raise ValueError(f"pdd_num_steps={steps} not divisible by ptd_num_hops={hops}")
    heads = {}
    for prefix, native in HEADS.items():
        if f"{prefix}.weight" not in tensors:
            raise ValueError(f"PTD file is missing the PDD head bank {prefix}.weight")
        w, b = tensors[f"{prefix}.weight"], tensors.get(f"{prefix}.bias")
        if w.ndim != 3 or w.shape[0] != steps or (b is not None and tuple(b.shape) != (steps, w.shape[1])):
            raise ValueError(f"PDD head bank {prefix} must be ({steps}, out, in)")
        heads[native] = (w, b)
    modules = sorted({k.rsplit(".lora_", 1)[0] for k in tensors if ".lora_" in k})
    bank = defaultdict(lambda: defaultdict(list))
    for module in modules:
        name, qkv = _target(module)
        down, up = tensors[f"{module}.lora_down"], tensors[f"{module}.lora_up"]
        sd, su = tensors.get(f"{module}.lora_down_scale"), tensors.get(f"{module}.lora_up_scale")
        if down.ndim != 3 or up.ndim != 3 or down.shape[0] != hops or up.shape[0] != hops:
            raise ValueError(f"{module}: expected factors stacked over {hops} hops")
        for h in range(hops):
            a, b = down[h], up[h]
            sa = None if sd is None else sd[h].float()
            sb = None if su is None else su[h].float()
            if name.endswith(".mlp.fc1"):  # Diffusers stores [up, gate]; native H3 consumes [gate, up]
                b = torch.cat(b.chunk(2, dim=0)[::-1], dim=0)
                sb = None if sb is None else torch.cat(sb.chunk(2, dim=0)[::-1], dim=0)
            if scale != 1.0:
                sb = torch.full((b.shape[0],), scale) if sb is None else sb * scale
            bank[name][h].append(expert_cls(a.contiguous(), b.contiguous(), sa, sb, qkv))
    expected = {_target(f"{p}.{i}.{part}")[0] for p, n in (("transformer_blocks", _count(modules, "transformer_blocks")),
                                                           ("token_refiner.refiner_blocks", _count(modules, "token_refiner")))
                for i in range(n) for part in PARTS}
    if set(bank) != expected or any(len(by_hop[h]) != (3 if n.endswith("qkv_proj") else 1)
                                    for n, by_hop in bank.items() for h in range(hops)):
        raise ValueError("Incomplete or unexpected PTD topology")
    return ({n: {h: tuple(e) for h, e in by_hop.items()} for n, by_hop in bank.items()}, heads, hops, steps,
            _count(modules, "transformer_blocks"), _count(modules, "token_refiner"))


def _count(modules, prefix):
    found = {int(_MODULE.fullmatch(m)[2]) for m in modules if m.startswith(prefix)}
    if found and found != set(range(len(found))):
        raise ValueError(f"PTD file has non-contiguous {prefix} indices")
    return len(found)


def to_int8(tensors, meta):
    """The compact version of a stacked-hop file: int8 factors with one float32 scale per row.

    ``lora_down`` [H, r, in] gets a scale per (hop, rank row); ``lora_up`` [H, out, r] one per (hop, out row).
    The PDD heads are small and stay as they are.
    """
    out = {}
    for key, value in tensors.items():
        if key.endswith((".lora_down", ".lora_up")):
            x = value.float()
            s = x.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127
            out[key] = (x / s).round().clamp(-127, 127).to(torch.int8)
            out[key + "_scale"] = s.squeeze(-1)
        else:
            out[key] = value
    meta = dict(meta, ptd_compact=json.dumps({"schema": 1, "quant": "int8 per-row"}))
    return out, meta


# ------------------------------------------------------------------------------------ controller

def controller_class(patcher, sampling, base_cls):
    """``base_cls`` (the compact controller) over a stacked-hop bank, this file's hop grid and PDD heads."""

    class PDDController(base_cls):
        def __init__(self, native, bank, heads, hops, steps, video_shift, audio_shift):
            self.recipe = f"pdd_{hops}hop"  # not an upstream recipe: the official Sampler/Sigmas refuse it
            self.hops = hops
            self.diffusion_model = native
            self._lock = patcher._model_lock(native)
            self._selection = ContextVar(f"ptd_selection_{id(self)}", default=None)
            self.interrupt_check = patcher._check_interrupt
            self._bank = bank
            self.sigmas = hop_sigmas(hops, steps, video_shift)
            corrections = audio_corrections(self.sigmas, sampling.audio_sigma, sampling.audio_sigma_prime)
            # Each hop's fused (weight, bias) per native head, computed once on the CPU.
            self._heads = []
            for h, (video_plan, audio_plan) in enumerate(hop_plans(hops, steps, video_shift, audio_shift)):
                fused = {}
                for name, (w, b) in heads.items():
                    plan = video_plan if name.endswith("video_out") else audio_plan
                    weight = torch.einsum("n,noi->oi", plan, w.float())
                    bias = None if b is None else torch.einsum("n,no->o", plan, b.float())
                    # Kept in float32 and cast to the head's compute dtype when applied (small: out x in).
                    fused[name] = (weight, bias, corrections[h] if name.endswith("audio_out") else 1.0)
                self._heads.append(fused)
            for name, (w, _) in heads.items():
                parent, _, attr = name.rpartition(".")
                head = getattr(native.get_submodule(parent), attr)
                if (head.in_features, head.out_features) != (w.shape[2], w.shape[1]):
                    raise ValueError(f"PDD head {name} is {tuple(w.shape[1:])} but this base's head is "
                                     f"({head.out_features}, {head.in_features})")

        @contextmanager
        def activate(self, hop, sigma):
            if isinstance(hop, bool) or not isinstance(hop, int) or not 0 <= hop < self.hops:
                raise ValueError(f"PTD hop must be in [0, {self.hops})")
            expected = float(self.sigmas[hop])
            if abs(float(sigma) - expected) > 2e-6:
                raise ValueError(f"Hop {hop} must run at its trained time {expected:.6f}, not {float(sigma):.6f}")
            token = self._selection.set(patcher._Selection(hop, expected))
            try:
                yield self
            finally:
                self._selection.reset(token)

        @contextmanager
        def forward_hooks(self, selection):
            # Registered before the official hooks, so the official audio clock then scales the fused head
            # by its tangent; the stored chord / tangent factor makes the net scale the chord.
            fused = self._heads[selection.hop]
            frame = []
            with ExitStack() as stack:
                for name, (weight, bias, factor) in fused.items():
                    parent, _, attr = name.rpartition(".")
                    head = getattr(self.diffusion_model.get_submodule(parent), attr)

                    def hook(module, args, output, weight=weight, bias=bias, factor=factor):
                        if not frame or patcher._FORWARD.get() is not frame[0]:
                            return output
                        x = args[0]
                        out = F.linear(x.to(output.dtype), weight.to(x.device, output.dtype),
                                       None if bias is None else bias.to(x.device, output.dtype))
                        return out if factor == 1.0 else out * factor

                    stack.callback(head.register_forward_hook(hook).remove)
                stack.enter_context(super().forward_hooks(selection))
                frame.append(patcher._FORWARD.get())
                yield

    return PDDController


def patch(model, tensors, meta, patcher, sampling, base_cls, expert_cls,
          video_shift=VIDEO_SHIFT, audio_shift=AUDIO_SHIFT):
    """A native MODEL clone carrying a stacked-hop PTD bank, its hop grid and its PDD heads."""
    if patcher.CONTROLLER_KEY in model.model_options:
        raise ValueError("This MODEL already has PTD adapters; start from the base MODEL")
    native = model.get_model_object("diffusion_model")
    required = ("blocks", "token_refiner", "condition_proj", "final_layer", "sigma_shift_video", "sigma_shift_audio")
    if not all(hasattr(native, name) for name in required):
        raise ValueError("PTD requires a native MiniMax-H3 MODEL")
    bank, heads, hops, steps, blocks, refiners = load(tensors, meta, expert_cls)
    if (blocks, refiners) != (len(native.blocks), len(native.token_refiner.blocks)):
        raise ValueError(f"PTD file is for {blocks}+{refiners} blocks, this base has "
                         f"{len(native.blocks)}+{len(native.token_refiner.blocks)}")
    for name, by_hop in bank.items():
        module = native.get_submodule(name)
        for e in by_hop[0]:
            if e.a.shape[1] != module.in_features or \
                    e.b.shape[0] * (3 if e.qkv_index is not None else 1) != module.out_features:
                raise ValueError(f"PTD adapter does not fit {name}")
    controller = controller_class(patcher, sampling, base_cls)(native, bank, heads, hops, steps, video_shift, audio_shift)

    clone = model.clone()
    # PTD carries physical audio latents (audio_scale=1 inside each call), so the sampler's audio carry is 1.
    model_sampling = copy.deepcopy(model.get_model_object("model_sampling"))
    if not hasattr(model_sampling, "audio_shift") or not hasattr(model_sampling, "shift"):
        raise ValueError("PTD requires the native MiniMax-H3 audio/video flow sampler")
    model_sampling.audio_shift = model_sampling.shift
    clone.add_object_patch("model_sampling", model_sampling)
    # Native H3 refines text once, outside any hop; keep raw embeddings so each hop's refiner adapters run.
    text_dim = native.condition_proj.in_features

    def raw_text(text_states):
        if text_states.shape[-1] != text_dim:
            raise ValueError("PTD requires raw MiniMax-H3 text conditioning, not pre-refined embeddings")
        return text_states

    clone.add_object_patch("diffusion_model.preprocess_text_embeds", raw_text)
    previous = clone.model_options.get("model_function_wrapper")
    clone.set_model_unet_function_wrapper(patcher._ModelFunctionWrapper(controller, previous))
    clone.model_options[patcher.CONTROLLER_KEY] = controller
    return clone


@torch.no_grad()
def sample(model, x, sigmas, extra_args=None, callback=None, disable=None, *, controller, sampling):
    """The official fixed-hop recurrence (ptd_comfy.sampling.sample_ptd) on this checkpoint's own grid."""
    del disable
    expected = controller.sigmas
    if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or sigmas.shape != expected.shape or \
            not torch.equal(sigmas.detach().cpu(), expected.to(sigmas.dtype)):
        raise ValueError(f"PTD requires this checkpoint's {controller.hops}-hop grid; use PTD Sigmas (compact)")
    extra_args = {} if extra_args is None else dict(extra_args)
    sampling._validate_initial_inputs(model, extra_args)
    raw_noise = getattr(model, "noise", None)
    state = sampling.initial_state(raw_noise) if raw_noise is not None else sampling._map(x, lambda v: v.float())
    sampling._pair(state, x, lambda a, b: a)
    first = sampling._streams(state)[0]
    grid = expected.to(device=first.device)
    for hop in range(len(grid) - 1):
        time = float(grid[hop])
        with controller.activate(hop, time):
            denoised = model(state, grid[hop].expand(first.shape[0]), **extra_args)
        velocity = sampling.denoised_to_velocity(state, denoised, time)
        if callback is not None:
            callback({"x": state, "i": hop, "sigma": grid[hop], "sigma_hat": grid[hop], "denoised": denoised})
        delta = (grid[hop + 1] - grid[hop]).float()
        state = sampling._pair(state, velocity, lambda value, flow, delta=delta: value.float() + delta * flow.float())
    return state

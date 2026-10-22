"""A tiny real native MiniMax-H3 in ComfyUI: compact (int8, variable-rank) PTD adapters stacked with an
ordinary LoRA loaded the way LoraLoaderModelOnly loads it, in both orders.

Run with ComfyUI and ComfyUI-PTD importable, e.g.
  PYTHONPATH=/path/to/ComfyUI:/path/to/ComfyUI-PTD:/path/to/custom_nodes python -m pytest tests -q
"""
import importlib
import os
import sys
import types

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

comfy_args = pytest.importorskip("comfy.cli_args")
comfy_args.args.cpu = True
import comfy.lora  # noqa: E402
import comfy.model_management  # noqa: E402
import comfy.model_patcher  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.supported_models  # noqa: E402

pytest.importorskip("ptd_comfy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
compact = importlib.import_module("ComfyUI-PTD-Compact.compact")
from ptd_comfy.checkpoint import adapter_metadata  # noqa: E402

DIMS = dict(hidden_size=64, num_layers=2, token_refiner_num_layers=1, num_attention_heads=2, attention_head_dim=32,
            ffn_hidden_size=48, text_dim=40, timestep_input_dim=16, time_embed_hidden_size=32, time_embed_dim=16,
            rope_inv_freq_len=4)
PARTS = {"attn.to_q": "attn.qkv_proj", "attn.to_k": "attn.qkv_proj", "attn.to_v": "attn.qkv_proj",
         "attn.to_out.0": "attn.out_proj", "ff.net.0.proj": "mlp.fc1", "ff.net.2": "mlp.fc2"}


def native_model(curve=False):
    """curve=True builds the pruned layout (adaln_t_table instead of time_embedder + full adaln_proj)."""
    extra = {"adaln_curve_grid": 9} if curve else {}
    config = comfy.supported_models.MiniMaxH3(dict(image_model="minimax_h3", **DIMS, **extra))
    config.set_inference_dtype(torch.float32, None)
    model = config.get_model({})
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.diffusion_model.parameters():
            p.normal_(0, 0.05)
        dm = model.diffusion_model
        n = dm.rope.inv_freq.numel()
        dm.rope.inv_freq.copy_(1.0 / 10000 ** (torch.arange(n, dtype=torch.float32) / n))
        if curve:
            dm.adaln_t_table.normal_(0, 0.5)
    cpu = torch.device("cpu")
    return comfy.model_patcher.ModelPatcher(model, cpu, cpu)


def adapter_file(path, native):
    """int8 + per-row scales, a different rank per module and hop (like the shrunk releases)."""
    g = torch.Generator().manual_seed(1)
    tensors, dense = {}, {}
    modules = [(f"transformer.transformer_blocks.{i}", f"blocks.{i}") for i in range(2)] + \
              [("transformer.token_refiner.refiner_blocks.0", "token_refiner.blocks.0")]
    for src, dst in modules:
        for part, native_part in PARTS.items():
            layer = native.get_submodule(f"{dst}.{native_part}")
            out = layer.out_features // (3 if native_part == "attn.qkv_proj" else 1)
            for hop in range(4):
                r = 3 + hop + len(part) % 3
                A = torch.randn(r, layer.in_features, generator=g) * 0.1
                B = torch.randn(out, r, generator=g) * 0.1
                key = f"{src}.{part}"
                for f, X in (("A", A), ("B", B)):
                    s = X.abs().amax(1, keepdim=True) / 127
                    tensors[f"{key}.lora_{f}.{hop}.weight"] = (X / s).round().to(torch.int8)
                    tensors[f"{key}.lora_{f}.{hop}.weight_scale"] = s
                qa = tensors[f"{key}.lora_A.{hop}.weight"].float() * tensors[f"{key}.lora_A.{hop}.weight_scale"]
                qb = tensors[f"{key}.lora_B.{hop}.weight"].float() * tensors[f"{key}.lora_B.{hop}.weight_scale"]
                dense[(f"{dst}.{native_part}", part, hop)] = qb @ qa
    for head, native_head in (("proj_out", "video_out"), ("audio_proj_out", "audio_out")):
        layer = native.final_layer.get_submodule(native_head)
        for hop in range(4):
            tensors[f"transformer.{head}.main.lora_A.{hop}.weight"] = torch.randn(layer.out_features, layer.in_features, generator=g) * 0.01
            tensors[f"transformer.{head}.main.lora_B.{hop}.weight"] = torch.eye(layer.out_features)
    meta = adapter_metadata("h3_4step")
    meta["ptd_format"] = compact.COMPACT
    meta["ptd_compact"] = '{"schema": 1}'
    save_file(tensors, path, metadata=meta)
    return dense


def lora_for(native):
    g = torch.Generator().manual_seed(2)
    sd, dense = {}, {}
    for name in ("blocks.0.attn.qkv_proj", "blocks.1.mlp.fc1", "blocks.1.mlp.fc2", "token_refiner.blocks.0.attn.out_proj"):
        layer = native.get_submodule(name)
        up, down = torch.randn(layer.out_features, 4, generator=g) * 0.1, torch.randn(4, layer.in_features, generator=g) * 0.1
        sd[f"{name}.lora_up.weight"], sd[f"{name}.lora_down.weight"] = up, down
        dense[name] = 0.7 * up @ down
    return sd, dense


def expected_delta(dense, name, hop):
    """The PTD delta for a native module (fused QKV stacked, fc1 rows [gate, up])."""
    parts = [(p, d) for (n, p, h), d in dense.items() if n == name and h == hop]
    if name.endswith("qkv_proj"):
        order = {"attn.to_q": 0, "attn.to_k": 1, "attn.to_v": 2}
        return torch.cat([d for _, d in sorted(parts, key=lambda pd: order[pd[0]])], 0)
    (part, d), = parts
    if part == "ff.net.0.proj":
        up, gate = d.chunk(2, 0)
        return torch.cat([gate, up], 0)
    return d


@pytest.mark.parametrize("curve", [False, True], ids=["full", "pruned"])
@pytest.mark.parametrize("lora_first", [True, False])
def test_compact_ptd_stacks_with_lora(tmp_path, lora_first, curve):
    base = native_model(curve)
    native = base.get_model_object("diffusion_model")
    weights = {n: p.detach().clone() for n, p in native.named_parameters()}
    ptd_dense = adapter_file(str(tmp_path / "a.safetensors"), native)
    lora_sd, lora_dense = lora_for(native)

    def add_lora(m):
        return comfy.sd.load_lora_for_models(m, None, lora_sd, 0.7, 0)[0]

    model = add_lora(base) if lora_first else base
    model = compact.patch_compact(model, str(tmp_path / "a.safetensors"))
    if not lora_first:
        model = add_lora(model)
    controller = model.model_options["ptd_native_controller"]
    assert native.use_adaln_curves == curve  # restored after the official checks
    comfy.model_management.load_models_gpu([model])
    try:
        x = torch.randn(5, DIMS["hidden_size"])
        for hop, sigma in enumerate((0.999, 0.947, 0.857, 0.667)):
            for name in ("blocks.0.attn.qkv_proj", "blocks.1.mlp.fc1", "blocks.1.mlp.fc2", "blocks.0.attn.out_proj",
                         "token_refiner.blocks.0.attn.out_proj", "token_refiner.blocks.0.attn.qkv_proj"):
                layer = native.get_submodule(name)
                inp = torch.randn(5, layer.in_features)
                W = weights[name + ".weight"] + lora_dense.get(name, 0)
                ref = F.linear(inp, W + expected_delta(ptd_dense, name, hop), weights.get(name + ".bias"))
                with controller.activate(hop, sigma), controller.forward_hooks(controller.current_selection()):
                    got = layer(inp)
                torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)
                # Outside a PTD call the layer is the LoRA'd base only.
                torch.testing.assert_close(layer(inp), F.linear(inp, W, weights.get(name + ".bias")), rtol=1e-4, atol=1e-5)
        bank = sum(t.numel() * t.element_size() for by_hop in controller._bank.values()
                   for e in by_hop.values() for x in e for t in (x.a, x.b))
        assert all(x.a.dtype == torch.int8 for by_hop in controller._bank.values() for e in by_hop.values()
                   for x in e if x.sa is not None)
        assert bank > 0
    finally:
        comfy.model_management.unload_all_models()
    # The base MODEL is untouched: no PTD controller, LoRA patches only on the LoRA clone.
    assert "ptd_native_controller" not in base.model_options
    assert not base.patches


def test_registered_official_pack_accepts_compact_model(tmp_path):
    """Load both packs the way ComfyUI does (path-named modules), then feed the compact loader's MODEL
    to the official PTD Sampler and PTD Sigmas nodes, whose isinstance check must accept it."""
    import asyncio
    import nodes

    official = os.environ.get("PTD_REPO") or os.path.dirname(os.path.dirname(sys.modules["ptd_comfy"].__file__))
    pack = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for path in (official, pack):
        assert asyncio.run(nodes.load_custom_node(path)), path
    sampler_cls = nodes.NODE_CLASS_MAPPINGS["PTDH3Sampler"]
    loader_cls = nodes.NODE_CLASS_MAPPINGS["PTDH3CompactLoader"]
    assert sampler_cls.__module__ != "ptd_comfy.nodes"  # the registered copy is path-named

    base = native_model()
    adapter_file(str(tmp_path / "a.safetensors"), base.get_model_object("diffusion_model"))
    import folder_paths
    folder_paths.add_model_folder_path("ptd", str(tmp_path))
    model, = loader_cls().load(base, "a.safetensors")
    sampler, = sampler_cls().get_sampler(model)
    sigmas, = nodes.NODE_CLASS_MAPPINGS["PTDH3Sigmas"]().get_sigmas(model)
    assert sigmas.tolist() == [0.999, 0.947, 0.857, 0.667, 0.0]
    assert sampler is not None


@pytest.mark.parametrize("curve", [False, True], ids=["full", "pruned"])
def test_end_to_end_ptd_sampling(tmp_path, curve):
    """Real native H3 forward through ComfyUI's guider with the PTD sampler/sigmas, on a full and a pruned
    (AdaLN-curve) tiny base. The adapters must change the result, and it must stay finite."""
    import comfy.nested_tensor
    import comfy.samplers
    from ptd_comfy.sampling import make_sampler, ptd_sigmas

    def run(path):
        base = native_model(curve)
        native = base.get_model_object("diffusion_model")
        adapter_file(str(path), native) if not path.exists() else None
        model = compact.patch_compact(base, str(path))
        controller = model.model_options["ptd_native_controller"]
        guider = comfy.samplers.CFGGuider(model)
        context = torch.randn(1, 6, DIMS["text_dim"], generator=torch.Generator().manual_seed(3))
        guider.set_conds([[context, {}]], [[context, {}]])
        guider.set_cfg(1.0)
        video, audio = torch.zeros(1, 24, 2, 4, 4), torch.zeros(1, 32, 2, 3)
        noise = comfy.nested_tensor.NestedTensor([torch.randn(video.shape, generator=torch.Generator().manual_seed(4)),
                                                  torch.randn(audio.shape, generator=torch.Generator().manual_seed(5))])
        try:
            out = guider.sample(noise, comfy.nested_tensor.NestedTensor([video, audio]),
                                make_sampler(controller, controller.recipe), ptd_sigmas(controller.recipe),
                                disable_pbar=True, seed=7)
        finally:
            comfy.model_management.unload_all_models()
        return [t.clone() for t in out.unbind()]

    adapted = run(tmp_path / "a.safetensors")
    # the same adapters with every A zeroed: plain base on PTD's grid
    from safetensors.torch import load_file
    t, meta = load_file(str(tmp_path / "a.safetensors")), compact.read(str(tmp_path / "a.safetensors"))[1]
    save_file({k: (torch.zeros_like(v) if ".lora_A." in k and v.dtype == torch.int8 else v) for k, v in t.items()},
              str(tmp_path / "zero.safetensors"), metadata=meta)
    plain = run(tmp_path / "zero.safetensors")
    for a, p in zip(adapted, plain):
        assert torch.isfinite(a).all()
        assert not torch.allclose(a, p)


def test_comfy_compiler_stays_on_unless_it_failed(tmp_path):
    """The allocation compiler stays on during hops; after a failure it is off only inside PTD calls."""
    from comfy.cli_args import args
    base = native_model()
    adapter_file(str(tmp_path / "a.safetensors"), base.get_model_object("diffusion_model"))
    controller = compact.patch_compact(base, str(tmp_path / "a.safetensors")).model_options["ptd_native_controller"]
    saved = args.disable_comfy_compiler, args.disable_cuda_graphs
    args.disable_comfy_compiler = args.disable_cuda_graphs = False
    try:
        with controller.activate(0, 0.999), controller.forward_hooks(controller.current_selection()):
            assert not args.disable_comfy_compiler and not args.disable_cuda_graphs
        controller.allocation_compiler = False
        with controller.activate(0, 0.999), controller.forward_hooks(controller.current_selection()):
            assert args.disable_comfy_compiler and args.disable_cuda_graphs
        assert not args.disable_comfy_compiler and not args.disable_cuda_graphs
    finally:
        args.disable_comfy_compiler, args.disable_cuda_graphs = saved


def test_compiler_failure_resamples_without_it():
    """An aimdo compile error resamples that run with the compiler off; other errors pass through."""
    controller, calls = types.SimpleNamespace(allocation_compiler=True), []

    def flaky(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
        calls.append(controller.allocation_compiler)
        if controller.allocation_compiler:
            raise RuntimeError("aimdo memory compile error")
        return x + 1

    out = compact._with_progress(flaky)(None, torch.zeros(1), torch.tensor([0.999, 0.5, 0.0]), disable=True,
                                        controller=controller)
    assert calls == [True, False] and float(out) == 1.0
    assert controller.allocation_compiler, "the next run tries the compiler again"

    def broken(*args, **kwargs):
        raise RuntimeError("something else")

    with pytest.raises(RuntimeError, match="something else"):
        compact._with_progress(broken)(None, torch.zeros(1), torch.tensor([0.999, 0.0]), disable=True,
                                       controller=types.SimpleNamespace(allocation_compiler=True))


def test_ptd_sampler_ticks_a_console_bar_per_hop(tmp_path, capsys):
    """Like ComfyUI's own samplers: one tqdm tick per hop, and the UI callback still sees every hop."""
    base = native_model()
    adapter_file(str(tmp_path / "a.safetensors"), base.get_model_object("diffusion_model"))
    controller = compact.patch_compact(base, str(tmp_path / "a.safetensors")).model_options["ptd_native_controller"]
    from ptd_comfy import sampling
    assert sampling.sample_ptd.ptd_progress
    compact._install_progress(compact.ptd_modules()[1])  # idempotent: not wrapped twice
    assert not getattr(sampling.sample_ptd.__wrapped__, "ptd_progress", False)
    grid = sampling.ptd_sigmas(controller.recipe)
    seen = []

    def fake(model, x, sigmas, extra_args=None, callback=None, disable=None, **_):
        for i in range(len(sigmas) - 1):
            callback({"i": i})
        return x

    run = compact._with_progress(fake)
    run(None, torch.zeros(1), grid, callback=lambda r: seen.append(r["i"]), disable=False)
    err = capsys.readouterr().err
    hops = len(grid) - 1
    assert seen == list(range(hops)) and f"{hops}/{hops} [" in err
    run(None, torch.zeros(1), grid, disable=True)
    assert capsys.readouterr().err == ""


def _sample(model, controller):
    import comfy.nested_tensor
    import comfy.samplers
    from ptd_comfy.sampling import make_sampler, ptd_sigmas
    guider = comfy.samplers.CFGGuider(model)
    context = torch.randn(1, 6, DIMS["text_dim"], generator=torch.Generator().manual_seed(3))
    guider.set_conds([[context, {}]], [[context, {}]])
    guider.set_cfg(1.0)
    video, audio = torch.zeros(1, 24, 2, 4, 4), torch.zeros(1, 32, 2, 3)
    noise = comfy.nested_tensor.NestedTensor([torch.randn(video.shape, generator=torch.Generator().manual_seed(4)),
                                              torch.randn(audio.shape, generator=torch.Generator().manual_seed(5))])
    try:
        out = guider.sample(noise, comfy.nested_tensor.NestedTensor([video, audio]),
                            make_sampler(controller, controller.recipe), ptd_sigmas(controller.recipe),
                            disable_pbar=True, seed=7)
    finally:
        comfy.model_management.unload_all_models()
    return [t.clone() for t in out.unbind()]


@pytest.mark.parametrize("curve", [False, True], ids=["full", "pruned"])
def test_fast_kernels_give_the_same_samples(tmp_path, curve):
    """In fp32 the native fused blocks differ from PTD's reference arithmetic only by rounding."""
    def run(path, mode):
        base = native_model(curve)
        if not path.exists():
            adapter_file(str(path), base.get_model_object("diffusion_model"))
        model = compact.patch_compact(base, str(path), mode)
        return _sample(model, model.model_options["ptd_native_controller"])

    path = tmp_path / "a.safetensors"
    exact, fast = run(path, "exact (PTD reference)"), run(path, "fast (native kernels)")
    for e, f in zip(exact, fast):
        assert torch.allclose(e, f, rtol=1e-4, atol=1e-4), float((e - f).abs().max())


def test_fast_kernels_keep_every_adapter(tmp_path):
    """One MLP under a hop: in fast mode fc1's and fc2's adapters apply exactly as in exact mode."""
    def mlp_out(mode, x):
        base = native_model()
        path = tmp_path / "a.safetensors"
        if not path.exists():
            adapter_file(str(path), base.get_model_object("diffusion_model"))
        model = compact.patch_compact(base, str(path), mode)
        controller = model.model_options["ptd_native_controller"]
        mlp = model.get_model_object("diffusion_model").blocks[1].mlp
        plain = mlp(x)
        with controller.activate(1, 0.947), controller.forward_hooks(controller.current_selection()):
            return mlp(x), plain

    x = torch.randn(10, DIMS["hidden_size"], generator=torch.Generator().manual_seed(9))
    exact, plain = mlp_out("exact (PTD reference)", x)
    fast, _ = mlp_out("fast (native kernels)", x)
    assert float((exact - plain).abs().max()) > 1e-2, "the adapters change the MLP"
    assert torch.equal(fast, exact)


def test_fast_hop_keeps_native_blocks_and_restores_state(tmp_path):
    """During a fast hop the blocks keep their native forward; afterwards the official code is as before."""
    base = native_model()
    adapter_file(str(tmp_path / "a.safetensors"), base.get_model_object("diffusion_model"))
    model = compact.patch_compact(base, str(tmp_path / "a.safetensors"), "fast (native kernels)")
    controller = model.model_options["ptd_native_controller"]
    _, patcher = compact.ptd_modules()
    scope = patcher._scope_forward
    native = model.get_model_object("diffusion_model")
    fc2 = native.blocks[0].mlp.fc2
    with controller.activate(0, 0.999), controller.forward_hooks(controller.current_selection()):
        assert patcher._scope_forward is scope
        assert "forward" not in vars(native.blocks[0]), "fast mode keeps the native block forward"
        assert fc2.comfy_force_forward and fc2._forward_hooks, "fc2 still runs eager for its adapter"
    assert not fc2.comfy_force_forward and not fc2._forward_hooks


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_in_place_residual_matches_ptd_bit_for_bit(dtype):
    """Same bits as the official out-of-place residual, without allocating a new stream tensor."""
    def official(state, update, gate, segments):  # ptd_comfy.patcher._residual
        result = torch.empty_like(state)
        for start, stop, row in segments:
            product = update[start:stop] * gate[row].to(state.dtype)
            result[start:stop] = state[start:stop] + product
        return result

    g = torch.Generator().manual_seed(0)
    state = torch.randn(300, 64, generator=g).to(dtype)
    update = torch.randn(300, 64, generator=g).to(dtype)
    gate = torch.randn(3, 64, generator=g)  # fp32 rows, as AdaLN gives them
    segments = [(0, 200, 0), (200, 260, 1), (260, 300, 2)]
    expected = official(state, update, gate, segments)
    stream = state.clone()
    out = compact._residual_in_place(stream, update, gate, segments)
    assert out.data_ptr() == stream.data_ptr()
    assert torch.equal(out, expected)


def test_loader_installs_the_in_place_residual(tmp_path):
    base = native_model()
    adapter_file(str(tmp_path / "a.safetensors"), base.get_model_object("diffusion_model"))
    compact.patch_compact(base, str(tmp_path / "a.safetensors"))
    assert compact.ptd_modules()[1]._residual is compact._residual_in_place


# ------------------------------------------------------------------ Ref2VA (taobao-content-ai) files

def ref2va_file(path, native, hops=4, steps=32, rank=8):
    """A random file in the taobao Ref2VA layout: factors stacked over hops, bf16, plus a PDD head bank."""
    g = torch.Generator().manual_seed(1)
    parts = {"attn.to_q": ("attn.qkv_proj", 3), "attn.to_k": ("attn.qkv_proj", 3), "attn.to_v": ("attn.qkv_proj", 3),
             "attn.to_out.0": ("attn.out_proj", 1), "ff.net.0.proj": ("mlp.fc1", 1), "ff.net.2": ("mlp.fc2", 1)}
    t = {}
    for prefix, native_prefix, n in (("transformer_blocks", "blocks", len(native.blocks)),
                                     ("token_refiner.refiner_blocks", "token_refiner.blocks", len(native.token_refiner.blocks))):
        for i in range(n):
            for part, (module, div) in parts.items():
                layer = native.get_submodule(f"{native_prefix}.{i}.{module}")
                t[f"{prefix}.{i}.{part}.lora_down"] = (torch.randn(hops, rank, layer.in_features, generator=g) * 0.1).bfloat16()
                t[f"{prefix}.{i}.{part}.lora_up"] = (torch.randn(hops, layer.out_features // div, rank, generator=g) * 0.1).bfloat16()
    for prefix, head in (("proj_out", native.final_layer.video_out), ("audio_proj_out", native.final_layer.audio_out)):
        w = head.weight.detach().float()
        t[f"{prefix}.weight"] = (w[None] + torch.randn(steps, *w.shape, generator=g) * 0.05).bfloat16()
        t[f"{prefix}.bias"] = (torch.randn(steps, w.shape[0], generator=g) * 0.05).bfloat16()
    save_file(t, path, metadata={"ptd_num_hops": str(hops), "pdd_num_steps": str(steps), "lora_rank": str(rank),
                                 "lora_alpha": f"{rank}.0", "format": "pt"})


def _sample_with(model):
    import comfy.nested_tensor
    import comfy.samplers
    guider = comfy.samplers.CFGGuider(model)
    context = torch.randn(1, 6, DIMS["text_dim"], generator=torch.Generator().manual_seed(3))
    guider.set_conds([[context, {}]], [[context, {}]])
    guider.set_cfg(1.0)
    video, audio = torch.zeros(1, 24, 2, 4, 4), torch.zeros(1, 32, 2, 3)
    noise = comfy.nested_tensor.NestedTensor([torch.randn(video.shape, generator=torch.Generator().manual_seed(4)),
                                              torch.randn(audio.shape, generator=torch.Generator().manual_seed(5))])
    try:
        out = guider.sample(noise, comfy.nested_tensor.NestedTensor([video, audio]), compact.make_sampler(model),
                            compact.sigmas(model), disable_pbar=True, seed=7)
    finally:
        comfy.model_management.unload_all_models()
    return [t.clone() for t in out.unbind()]


@pytest.mark.parametrize("hops", [4, 8])
def test_ref2va_hop_grid(hops):
    """One knot per hop on the T_MAX clock, descending to 0; each hop's head plan averages its block."""
    sig = compact.pdd.hop_sigmas(hops)
    assert sig.shape == (hops + 1,) and abs(float(sig[0]) - compact.pdd.T_MAX) < 1e-12 and float(sig[-1]) == 0.0
    assert bool((sig.diff() < 0).all())
    for video, audio in compact.pdd.hop_plans(hops):
        for plan in (video, audio):
            assert abs(float(plan.sum()) - 1) < 1e-6 and int((plan > 0).sum()) == 32 // hops


@pytest.mark.parametrize("curve", [False, True], ids=["full", "pruned"])
def test_ref2va_samples_on_full_and_pruned_bases(tmp_path, curve):
    """The Ref2VA path loads on either base; exact and fast agree; int8 stays close to bf16."""
    path, qpath = tmp_path / "ref2va.safetensors", tmp_path / "ref2va_int8.safetensors"
    ref2va_file(str(path), native_model(curve).get_model_object("diffusion_model"))
    qtensors, qmeta = compact.pdd.to_int8(*compact.read(str(path)))
    save_file(qtensors, str(qpath), metadata=qmeta)
    exact = _sample_with(compact.patch_compact(native_model(curve), str(path)))
    fast = _sample_with(compact.patch_compact(native_model(curve), str(path), "fast (native kernels)"))
    q = _sample_with(compact.patch_compact(native_model(curve), str(qpath)))
    for e, f, i in zip(exact, fast, q):
        assert torch.isfinite(e).all()
        assert torch.allclose(e, f, rtol=1e-4, atol=1e-4)
        assert torch.allclose(e, i, rtol=1e-2, atol=1e-2)


def test_ref2va_heads_replace_the_native_heads(tmp_path):
    """During a hop the video head returns the hop's fused PDD head, not the base head."""
    base = native_model()
    ref2va_file(str(tmp_path / "r.safetensors"), base.get_model_object("diffusion_model"))
    model = compact.patch_compact(base, str(tmp_path / "r.safetensors"))
    controller = model.model_options["ptd_native_controller"]
    head = model.get_model_object("diffusion_model").final_layer.video_out
    x = torch.randn(5, head.in_features)
    plain = head(x)
    with controller.activate(1, float(controller.sigmas[1])), controller.forward_hooks(controller.current_selection()):
        hopped = head(x)
    weight, bias, _ = controller._heads[1]["final_layer.video_out"]
    assert torch.allclose(hopped, torch.nn.functional.linear(x, weight, bias), atol=1e-5)
    assert not torch.allclose(hopped, plain, atol=1e-3)
    assert torch.equal(head(x), plain), "the hook is gone after the hop"


def test_compact_sampler_and_sigmas_cover_both_formats(tmp_path):
    """Ref2VA gets its own grid; fl2va files get the official recipe grid and sampler."""
    base = native_model()
    ref2va_file(str(tmp_path / "r.safetensors"), base.get_model_object("diffusion_model"))
    r = compact.patch_compact(base, str(tmp_path / "r.safetensors"))
    assert torch.equal(compact.sigmas(r), compact.pdd.hop_sigmas(4).float())
    base = native_model()
    adapter_file(str(tmp_path / "a.safetensors"), base.get_model_object("diffusion_model"))
    e = compact.patch_compact(base, str(tmp_path / "a.safetensors"))
    from ptd_comfy.sampling import ptd_sigmas
    assert torch.equal(compact.sigmas(e), ptd_sigmas("h3_4step"))
    assert compact.make_sampler(e).sampler_function.ptd_progress


def test_official_sampler_and_sigmas_take_ref2va_models(tmp_path):
    """After the compact loader runs, the official PTD Sampler / Sigmas nodes also handle Ref2VA MODELs."""
    from ptd_comfy import nodes as official
    base = native_model()
    ref2va_file(str(tmp_path / "r.safetensors"), base.get_model_object("diffusion_model"))
    model = compact.patch_compact(base, str(tmp_path / "r.safetensors"))
    (grid,) = official.PTDH3Sigmas().get_sigmas(model)
    assert torch.equal(grid, compact.pdd.hop_sigmas(4).float())
    (sampler,) = official.PTDH3Sampler().get_sampler(model)
    assert sampler.extra_options["controller"] is model.model_options["ptd_native_controller"]


# ------------------------------------------------------------------ progressive (SelfLift) sampling

progressive = importlib.import_module("ComfyUI-PTD-Compact.progressive")
CONTEXT = lambda: [[torch.randn(1, 6, DIMS["text_dim"], generator=torch.Generator().manual_seed(3)), {}]]  # noqa: E731


def _ref2va_model(tmp_path, hops=4):
    base = native_model()
    ref2va_file(str(tmp_path / f"r{hops}.safetensors"), base.get_model_object("diffusion_model"), hops=hops)
    return compact.patch_compact(base, str(tmp_path / f"r{hops}.safetensors"))


def _latent(h=4, w=4):
    import comfy.nested_tensor
    return {"samples": comfy.nested_tensor.NestedTensor([torch.zeros(1, 24, 2, h, w), torch.zeros(1, 32, 2, 3)])}


def _recording(monkeypatch):
    """Record every call of the PDD sampler: its grid, (flat-packed) latent shape and options."""
    calls, original = [], compact._pdd_sample

    def run(model, x, sigmas, **kwargs):
        calls.append({"sigmas": sigmas.clone(), "shape": tuple(x.shape), **kwargs})
        return original(model, x, sigmas, **kwargs)

    monkeypatch.setattr(compact, "_pdd_sample", run)
    return calls


def test_split_hops_equal_one_run(tmp_path):
    """Hops 0..k-1 then k..N-1 from the carried state give the same sample as one run."""
    import comfy.nested_tensor
    import comfy.samplers
    model = _ref2va_model(tmp_path, hops=4)
    controller = model.model_options["ptd_native_controller"]
    sampling, grid = compact._sampling(compact.ptd_modules()[1]), compact.sigmas(model)
    latent = _latent()["samples"]
    noise = comfy.nested_tensor.NestedTensor([torch.randn(t.shape, generator=torch.Generator().manual_seed(4 + i))
                                              for i, t in enumerate(latent.unbind())])
    zeros = comfy.nested_tensor.NestedTensor([torch.zeros_like(t) for t in latent.unbind()])

    def run(noise, sigmas, **options):
        guider = comfy.samplers.CFGGuider(model)
        guider.set_conds(CONTEXT(), CONTEXT())
        guider.set_cfg(1.0)
        sampler = comfy.samplers.KSAMPLER(compact._pdd_sample, extra_options={
            "controller": controller, "sampling": sampling, **options})
        return guider.sample(noise, latent, sampler, sigmas, disable_pbar=True, seed=7)

    try:
        whole = run(noise, grid)
        capture = {}
        run(noise, grid[:3], hops=(0, 2), capture=capture)
        split = run(zeros, grid[2:], hops=(2, 4), start=capture["next"])
    finally:
        comfy.model_management.unload_all_models()
    for a, b in zip(whole.unbind(), split.unbind()):
        assert torch.allclose(a, b, rtol=1e-6, atol=1e-6), float((a - b).abs().max())


def test_partial_hops_are_refused_without_a_start(tmp_path):
    model = _ref2va_model(tmp_path)
    controller = model.model_options["ptd_native_controller"]
    sampling = compact._sampling(compact.ptd_modules()[1])
    with pytest.raises(ValueError, match="state entering"):
        compact.pdd.sample(None, _latent()["samples"], compact.sigmas(model)[1:], controller=controller,
                           sampling=sampling, hops=(1, 4))
    with pytest.raises(ValueError, match="hop grid"):  # the plain sampler still wants the whole grid
        compact.pdd.sample(None, _latent()["samples"], compact.sigmas(model)[:3], controller=controller,
                           sampling=sampling)


@pytest.mark.parametrize("hops,k", [(4, 2), (8, 6)])
def test_progressive_runs_early_hops_small_and_renoises_the_lift(tmp_path, monkeypatch, hops, k):
    """Hops 0..k-1 run on the half-size latent, k..N-1 at full size from t·n' + (1-t)·lift(x0), the audio carried."""
    import comfy.sample
    import comfy.utils
    calls = _recording(monkeypatch)
    model = _ref2va_model(tmp_path, hops)
    latent_format = model.get_model_object("latent_format")
    lifted = []

    def lift(z0_low, out_hw):  # nearest, in the VAE's space
        z = torch.nn.functional.interpolate(z0_low.float(), size=(z0_low.shape[2], *out_hw), mode="nearest")
        lifted.append(z)
        return z, None

    try:
        out = progressive.sample(model, CONTEXT(), _latent(), seed=11, transition_hop=k, lowres_scale=0.5, lift=lift)
    finally:
        comfy.model_management.unload_all_models()
    grid = compact.sigmas(model)
    low, high = calls
    audio_size = 32 * 2 * 3  # samplers see video + audio as one flat [B, 1, N] pack
    assert low["hops"] == (0, k) and low["shape"] == (1, 1, 24 * 2 * 2 * 2 + audio_size)
    assert high["hops"] == (k, hops) and high["shape"] == (1, 1, 24 * 2 * 4 * 4 + audio_size)
    assert torch.equal(low["sigmas"], grid[:k + 1]) and torch.equal(high["sigmas"], grid[k:])
    video, audio = out["samples"].unbind()
    assert video.shape == (1, 24, 2, 4, 4) and audio.shape == (1, 32, 2, 3)
    assert torch.isfinite(video).all() and torch.isfinite(audio).all()

    low_shapes = [(1, 24, 2, 2, 2), (1, 32, 2, 3)]
    start = comfy.utils.unpack_latents(high["start"], [(1, 24, 2, 4, 4), (1, 32, 2, 3)])
    carried = comfy.utils.unpack_latents(low["capture"]["next"], low_shapes)
    t = float(grid[k])
    z0 = latent_format.process_in(lifted[0])
    fresh = comfy.sample.prepare_noise(z0, 12, None).to(z0)
    assert torch.allclose(start[0], t * fresh + (1 - t) * z0, atol=1e-5)
    assert torch.equal(start[1], carried[1].float())


def test_progressive_refuses_bad_settings(tmp_path):
    model = _ref2va_model(tmp_path, hops=4)
    lift = lambda z, hw: (z, None)  # noqa: E731
    with pytest.raises(ValueError, match="transition_hop"):
        progressive.sample(model, CONTEXT(), _latent(), 0, 4, 0.5, lift)
    masked = {**_latent(), "noise_mask": torch.ones(1, 4, 4)}
    with pytest.raises(ValueError, match="noise masks"):
        progressive.sample(model, CONTEXT(), masked, 0, 2, 0.5, lift)
    base = native_model()
    adapter_file(str(tmp_path / "a.safetensors"), base.get_model_object("diffusion_model"))
    with pytest.raises(ValueError, match="Ref2VA"):
        progressive.sample(compact.patch_compact(base, str(tmp_path / "a.safetensors")), CONTEXT(), _latent(), 0, 2,
                           0.5, lift)


class _PixelVAE:
    """Stand-in video VAE: 2x pixels per latent cell, first three channels as RGB."""
    vae_dtype, device = torch.float32, torch.device("cpu")

    def decode(self, z):
        frames = z[0, :3].permute(1, 2, 3, 0)
        return frames.repeat_interleave(2, 1).repeat_interleave(2, 2)

    def encode(self, frames):
        x = torch.nn.functional.avg_pool2d(frames.movedim(-1, 1).float(), 2)  # [T, 3, H, W]
        return x.permute(1, 0, 2, 3)[None].repeat(1, 8, 1, 1, 1)


@pytest.mark.skipif(not os.environ.get("PTD_SELFLIFT_DIR"), reason="set PTD_SELFLIFT_DIR to a comfyui-SelfLift checkout")
@pytest.mark.parametrize("rho", [0.0, 0.6, 1.0])
def test_node_with_real_selflift(tmp_path, rho):
    """The node end to end with SelfLift's own lift and correction (nearest, pixel anchor, pure anchor)."""
    nodes = importlib.import_module("ComfyUI-PTD-Compact.nodes")
    model = _ref2va_model(tmp_path, hops=4)
    try:
        (out,) = nodes.PTDH3SelfLiftSampler().sample(model, CONTEXT(), _PixelVAE(), _latent(), 3, 2, 0.5, rho, 1.0, 1.0,
                                                     "none")
    finally:
        comfy.model_management.unload_all_models()
    video, audio = out["samples"].unbind()
    assert video.shape == (1, 24, 2, 4, 4) and torch.isfinite(video).all() and torch.isfinite(audio).all()


def test_memory_is_reset_before_the_first_hop(tmp_path, monkeypatch):
    """The watermark / cache reset runs before hop 0 too (a SelfLift transition precedes it in the same node)."""
    seen = []
    monkeypatch.setattr(compact, "_between_hops", lambda model, hop: seen.append(hop))
    _sample_with(_ref2va_model(tmp_path))
    assert seen == ["start", 0, 1, 2, 3]


@pytest.mark.skipif(not os.environ.get("PTD_SELFLIFT_DIR"), reason="set PTD_SELFLIFT_DIR to a comfyui-SelfLift checkout")
@pytest.mark.parametrize("rho,unloaded", [(0.0, False), (0.6, True)])
def test_vae_is_unloaded_after_the_pixel_anchor(tmp_path, monkeypatch, rho, unloaded):
    import comfy.model_management
    calls = []
    monkeypatch.setattr(comfy.model_management, "unload_model_and_clones",
                        lambda patcher, **kw: calls.append(patcher))
    vae = _PixelVAE()
    vae.patcher = object()
    lift, combine = progressive.selflift_lift(vae, rho, 1.0, 1.0)
    lift(torch.randn(1, 24, 2, 2, 2), (4, 4))
    assert calls == ([vae.patcher] if unloaded else [])

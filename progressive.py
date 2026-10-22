"""Progressive-resolution PTD sampling with SelfLift (experimental): the first hops at low resolution, the rest
at full resolution, for the Ref2VA (PDD) files.

SelfLift (arXiv:2609.02036, ComfyUI port github.com/facok/comfyui-SelfLift) runs the early Euler steps of a
rectified-flow model on a smaller latent, predicts the clean video at the transition, lifts it to full size and
re-noises it there. A Ref2VA PTD file samples with exactly that kind of step (x0 per hop, ``state + Δt·v`` on
a fixed grid), so the split is the same, with every hop still run at its own trained time and adapter:

1. hops ``0 .. k-1`` on a ``lowres_scale`` latent; hop k-1's x0 is the transition prediction;
2. that video x0 is lifted to full size with SelfLift's own functions (nearest lift, its pixel-VAE anchor
   ``rho``, or the external H3 latent upscaler), imported from an installed comfyui-SelfLift;
3. the lifted x0 is re-noised at hop k's time with fresh noise, ``t·n + (1-t)·x0`` on PTD's clock;
   the audio keeps the state the low-res hops produced (it has no resolution);
4. hops ``k .. N-1`` at full size.

The hop count is unchanged. Neither H3 nor the adapters were trained at the low resolution, so this is a
speed experiment; compare against a plain PTD render on the same seed.
"""
from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import sys

import torch

from . import compact

_PACKAGE = "_ptd_compact_selflift"


def selflift_modules():
    """(selflift, h3_upscaler) of the installed comfyui-SelfLift, or a clear error."""
    try:
        import nodes
        node = nodes.NODE_CLASS_MAPPINGS.get("SelfLiftH3Sampler")
    except ImportError:  # outside ComfyUI (tests)
        node = None
    if node is not None:
        package = node.__module__.rsplit(".", 1)[0]
        return importlib.import_module(package + ".selflift"), importlib.import_module(package + ".h3_upscaler")
    # Not registered (yet): import the checkout next to this pack as a package of its own.
    custom_nodes = os.environ.get("PTD_SELFLIFT_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [custom_nodes] if os.path.isfile(os.path.join(custom_nodes, "selflift.py")) else \
        [os.path.join(custom_nodes, d) for d in sorted(os.listdir(custom_nodes))]
    for path in candidates:
        if os.path.isfile(os.path.join(path, "selflift.py")) and os.path.isfile(os.path.join(path, "h3_upscaler.py")):
            if _PACKAGE not in sys.modules:
                spec = importlib.util.spec_from_file_location(_PACKAGE, os.path.join(path, "__init__.py"),
                                                              submodule_search_locations=[path])
                sys.modules[_PACKAGE] = importlib.util.module_from_spec(spec)  # submodules only, no node registration
            return importlib.import_module(_PACKAGE + ".selflift"), importlib.import_module(_PACKAGE + ".h3_upscaler")
    raise RuntimeError("PTD SelfLift needs comfyui-SelfLift: run `git clone https://github.com/facok/comfyui-SelfLift` "
                       "inside ComfyUI/custom_nodes and restart ComfyUI.")


def upscaler_choices():
    try:
        return ["none"] + selflift_modules()[1].list_upscaler_models()
    except Exception:  # SelfLift missing: the node still loads and explains at run time
        return ["none"]


def selflift_lift(vae, rho, w_min, w_max, upscaler_model="none", upscaler_unload=True):
    """``lift(z0_low, (H, W))`` in the VAE's latent space, as SelfLift's H3 sampler does it."""
    if rho == 0.0 and upscaler_model == "none":
        logging.info("PTD SelfLift: rho=0 and no upscaler, so the lift is plain nearest-neighbour")
    selflift, upscaler = selflift_modules()
    lifter = None
    if upscaler_model != "none":
        lifter = lambda z, hw: upscaler.learned_latent_lift(z, hw, upscaler_model, force_unload=upscaler_unload)  # noqa: E731
    need_pix = rho > 0.0 and w_max > 0.0
    need_lat = not (rho >= 1.0 and w_min >= 1.0 and w_max >= 1.0)

    def lift(z0_low, out_hw):
        z_lat, z_pix = selflift.paired_lifts(z0_low, vae, out_hw, "nearest", lifter, need_lat=need_lat, need_pix=need_pix)
        return z_lat, z_pix

    def combine(z_lat, z_pix):
        return selflift.artifact_aware_consistency_lift(z_lat, z_pix, rho, w_min, w_max)

    return lift, combine


def sample(model, positive, latent_image, seed, transition_hop, lowres_scale, lift, combine=None):
    """Progressive PTD sampling; ``lift(z0_low, (H, W)) -> (z_lat, z_pix)`` works in the VAE's latent space and
    ``combine(z_lat, z_pix)`` (in model space) picks the full-size x0. Returns the output LATENT dict."""
    import comfy.nested_tensor
    import comfy.sample
    import comfy.samplers
    import comfy.model_management
    import comfy.utils
    import latent_preview
    from comfy.samplers import KSAMPLER

    _, patcher = compact.ptd_modules()
    controller = patcher.get_ptd_controller(model)
    if not hasattr(controller, "sigmas"):
        raise ValueError("PTD SelfLift supports the Ref2VA (PDD) files only")
    hops = controller.hops
    if not 1 <= transition_hop <= hops - 1:
        raise ValueError(f"transition_hop must be 1..{hops - 1} for this {hops}-hop file")
    if not 0.25 <= lowres_scale <= 1.0:
        raise ValueError("lowres_scale must be between 0.25 and 1")
    if latent_image.get("noise_mask") is not None:
        raise ValueError("PTD SelfLift does not take noise masks")

    samples = comfy.sample.fix_empty_latent_channels(model, latent_image["samples"],
                                                     latent_image.get("downscale_ratio_spacial", None),
                                                     latent_image.get("downscale_ratio_temporal", None))
    nested = samples.is_nested
    streams = list(samples.unbind()) if nested else [samples]
    if streams[0].ndim != 5:
        raise ValueError("PTD SelfLift expects a MiniMax-H3 video latent")
    b, c, t, H, W = streams[0].shape
    h, w = max(2, round(H * lowres_scale / 2) * 2), max(2, round(W * lowres_scale / 2) * 2)
    pack = (lambda s: comfy.nested_tensor.NestedTensor(s)) if nested else (lambda s: s[0])
    device = comfy.model_management.intermediate_device()
    low_latent = pack([torch.zeros(b, c, t, h, w, dtype=streams[0].dtype, device=device)] +
                      [torch.zeros_like(s, device=device) for s in streams[1:]])
    full_latent = pack([torch.zeros_like(s, device=device) for s in streams])

    grid = controller.sigmas.float()
    sampling = compact._sampling(patcher)
    run = compact._pdd_sample
    logging.info("PTD SelfLift: hops 0-%d at latent %dx%d, hops %d-%d at %dx%d (t=%.4f)",
                 transition_hop - 1, w, h, transition_hop, hops - 1, W, H, float(grid[transition_hop]))

    guider = comfy.samplers.CFGGuider(model)
    guider.set_conds(positive, positive)
    guider.set_cfg(1.0)
    preview = latent_preview.prepare_callback(model, hops)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    # 1. low-resolution hops; the last one's x0 is the transition prediction
    capture = {}
    noise = comfy.sample.prepare_noise(low_latent, seed, latent_image.get("batch_index", None))
    low = KSAMPLER(run, extra_options={"controller": controller, "sampling": sampling,
                                       "hops": (0, transition_hop), "capture": capture})
    guider.sample(noise, low_latent, low, grid[:transition_hop + 1],
                  callback=lambda i, x0, x, n: preview(i, x0, x, hops), disable_pbar=disable_pbar, seed=seed)

    # 2-3. lift the video x0, re-noise it at hop k's time; carry the audio state. Samplers see a nested latent
    # as one flat [B, 1, N] pack (CFGGuider.sample), so the captured tensors are unpacked here and the start
    # state is packed again.
    unpack = lambda v, shapes: comfy.utils.unpack_latents(v, shapes) if nested else [v]  # noqa: E731
    low_shapes = [s.shape for s in sampling._streams(low_latent)]
    x0 = unpack(capture["denoised"], low_shapes)
    carried = unpack(capture["next"], low_shapes)
    latent_format = model.get_model_object("latent_format")
    z_lat, z_pix = lift(latent_format.process_out(x0[0].float()).to(device), (H, W))
    z_lat = None if z_lat is None else latent_format.process_in(z_lat).to(device)
    z_pix = None if z_pix is None else latent_format.process_in(z_pix).to(device)
    z0 = combine(z_lat, z_pix) if combine is not None else z_lat
    fresh = comfy.sample.prepare_noise(z0, (seed + 1) % (1 << 64), latent_image.get("batch_index", None)).to(z0)
    time = float(grid[transition_hop])
    start = [time * fresh.float() + (1.0 - time) * z0.float()] + [s.to(device).float() for s in carried[1:]]
    start = comfy.utils.pack_latents(start)[0] if nested else start[0]
    del capture, x0, carried, z_lat, z_pix, z0, fresh

    # 4. full-resolution hops from the lifted state
    high = KSAMPLER(run, extra_options={"controller": controller, "sampling": sampling,
                                        "hops": (transition_hop, hops), "start": start})
    out = guider.sample(pack([torch.zeros_like(s) for s in streams]), full_latent, high,
                        grid[transition_hop:],
                        callback=lambda i, x0, x, n: preview(i + transition_hop, x0, x, hops),
                        disable_pbar=disable_pbar, seed=seed)
    result = latent_image.copy()
    result.pop("noise_mask", None)
    result["samples"] = out.to(device=comfy.model_management.intermediate_device(),
                               dtype=comfy.model_management.intermediate_dtype())
    return result

"""Loader, sampler and sigmas nodes for PTD MiniMax-H3 adapters: the EPFL fl2va files (standard or compact
int8 / reduced rank) and the taobao-content-ai Ref2VA files (4 / 8 hop, bf16 or compact int8)."""
from pathlib import Path

import folder_paths

from .compact import KERNELS, make_sampler, patch_compact, sigmas

if "ptd" not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("ptd", str(Path(folder_paths.models_dir) / "ptd"))


class PTDH3CompactLoader:
    CATEGORY = "PTD"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    DESCRIPTION = ("Apply PTD hop adapters (standard or compact int8 / reduced rank) to a native MiniMax-H3 model, "
                   "pruned or not: fl2va (EPFL) or Ref2VA (taobao-content-ai). Use PTD Sampler / Sigmas (compact).")

    @classmethod
    def INPUT_TYPES(cls):
        files = [f for f in folder_paths.get_filename_list("ptd") if f.endswith(".safetensors")]
        return {"required": {"model": ("MODEL",), "checkpoint": (files,)},
                "optional": {"kernels": (list(KERNELS), {"default": "exact (PTD reference)", "tooltip": (
                    "exact: PTD's reference block arithmetic, as the adapters were trained. fast: ComfyUI's fused "
                    "native blocks, ~8% faster per block. Same maths; only bf16 rounding differs.")})}}

    def load(self, model, checkpoint, kernels="exact (PTD reference)"):
        return (patch_compact(model, folder_paths.get_full_path_or_raise("ptd", checkpoint), kernels),)


class PTDH3CompactSampler:
    CATEGORY = "PTD"
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    DESCRIPTION = "PTD's fixed-hop sampler for any file the compact loader takes, Ref2VA (4 / 8 hop) included."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    def get_sampler(self, model):
        return (make_sampler(model),)


class PTDH3CompactSigmas:
    CATEGORY = "PTD"
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    DESCRIPTION = "The loaded PTD file's trained hop times. Not adjustable: PTD is not a scheduler."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    def get_sigmas(self, model):
        return (sigmas(model),)


class PTDH3SelfLiftSampler:
    CATEGORY = "PTD"
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    DESCRIPTION = ("Experimental: SelfLift progressive resolution for Ref2VA PTD files. The first hops run on a "
                   "smaller latent, the transition x0 is lifted with comfyui-SelfLift (must be installed) and the "
                   "remaining hops run at full size. Same hop count, faster early hops; quality is untested.")

    @classmethod
    def INPUT_TYPES(cls):
        from .progressive import upscaler_choices
        upscalers = upscaler_choices()
        h3 = [name for name in upscalers[1:] if "h3" in name.lower()]
        return {"required": {
            "model": ("MODEL", {"tooltip": "A Ref2VA model from Load PTD MiniMax-H3 Adapters (compact)."}),
            "positive": ("CONDITIONING",),
            "vae": ("VAE", {"tooltip": "Video VAE, for SelfLift's pixel anchor (rho > 0)."}),
            "latent_image": ("LATENT", {"tooltip": "The empty full-size AV latent (MiniMax H3 Reference to Video)."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "transition_hop": ("INT", {"default": 4, "min": 1, "max": 31, "tooltip": (
                "Hops run at low resolution, 1 .. hops-1. The rest run at full size.")}),
            "lowres_scale": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 1.0, "step": 0.05}),
            "rho": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": (
                "SelfLift: share of the most inconsistent locations corrected toward the pixel-VAE anchor (one VAE "
                "decode + encode). 0 skips it. SelfLift's H3 starting point: 0.6 with w_min = w_max = 1.")}),
            "w_min": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "w_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "upscaler_model": (upscalers, {"default": h3[0] if h3 else "none", "tooltip": (
                "SelfLift's external H3 latent upscaler, or none for a nearest-neighbour lift.")}),
        }, "optional": {
            "upscaler_unload": ("BOOLEAN", {"default": True}),
        }}

    def sample(self, model, positive, vae, latent_image, seed, transition_hop, lowres_scale, rho, w_min, w_max,
               upscaler_model, upscaler_unload=True):
        from . import progressive
        if not 0.0 <= w_min <= w_max <= 1.0:
            raise ValueError("w_min must not exceed w_max")
        lift, combine = progressive.selflift_lift(vae, rho, w_min, w_max, upscaler_model, upscaler_unload)
        return (progressive.sample(model, positive, latent_image, seed, transition_hop, lowres_scale, lift, combine),)


NODE_CLASS_MAPPINGS = {"PTDH3CompactLoader": PTDH3CompactLoader, "PTDH3CompactSampler": PTDH3CompactSampler,
                       "PTDH3CompactSigmas": PTDH3CompactSigmas, "PTDH3SelfLiftSampler": PTDH3SelfLiftSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"PTDH3CompactLoader": "Load PTD MiniMax-H3 Adapters (compact)",
                              "PTDH3CompactSampler": "PTD Sampler (compact)",
                              "PTDH3CompactSigmas": "PTD Sigmas (compact)",
                              "PTDH3SelfLiftSampler": "PTD SelfLift Sampler (compact, experimental)"}

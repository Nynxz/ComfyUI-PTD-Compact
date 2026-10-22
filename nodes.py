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


NODE_CLASS_MAPPINGS = {"PTDH3CompactLoader": PTDH3CompactLoader, "PTDH3CompactSampler": PTDH3CompactSampler,
                       "PTDH3CompactSigmas": PTDH3CompactSigmas}
NODE_DISPLAY_NAME_MAPPINGS = {"PTDH3CompactLoader": "Load PTD MiniMax-H3 Adapters (compact)",
                              "PTDH3CompactSampler": "PTD Sampler (compact)",
                              "PTDH3CompactSigmas": "PTD Sigmas (compact)"}

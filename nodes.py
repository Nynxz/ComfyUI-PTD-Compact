"""Loader node for compact PTD adapter files (int8 / reduced rank). Sampling uses the official
PTD Sampler and PTD Sigmas nodes from ComfyUI-PTD unchanged."""
from pathlib import Path

import folder_paths

from .compact import KERNELS, patch_compact

if "ptd" not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("ptd", str(Path(folder_paths.models_dir) / "ptd"))


class PTDH3CompactLoader:
    CATEGORY = "PTD"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    DESCRIPTION = ("Apply compact (int8 / reduced-rank) or standard PTD hop adapters to an unpruned native "
                   "MiniMax-H3 model. Use PTD Sampler and PTD Sigmas.")

    @classmethod
    def INPUT_TYPES(cls):
        files = [f for f in folder_paths.get_filename_list("ptd") if f.endswith(".safetensors")]
        return {"required": {"model": ("MODEL",), "checkpoint": (files,)},
                "optional": {"kernels": (list(KERNELS), {"default": "exact (PTD reference)", "tooltip": (
                    "exact: PTD's reference block arithmetic, as the adapters were trained. fast: ComfyUI's fused "
                    "native blocks, ~8% faster per block. Same maths; only bf16 rounding differs.")})}}

    def load(self, model, checkpoint, kernels="exact (PTD reference)"):
        return (patch_compact(model, folder_paths.get_full_path_or_raise("ptd", checkpoint), kernels),)


NODE_CLASS_MAPPINGS = {"PTDH3CompactLoader": PTDH3CompactLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"PTDH3CompactLoader": "Load PTD MiniMax-H3 Adapters (compact)"}

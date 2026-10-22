# ComfyUI-PTD-Compact

A loader, sampler and sigmas node for MiniMax-H3 PTD few-step adapters in ComfyUI. It loads both PTD
releases, in their original form or as smaller int8 files, on the full or the pruned base:

| Release | Task | Steps | Adapter files |
|---|---|---|---|
| EPFL ([epfl-vita/PTD-weights](https://huggingface.co/epfl-vita/PTD-weights)) | fl2va | 4, 6 | bf16 / int8 / int8 r5 in [nynxz/MiniMax-H3-PTD-LoRAs](https://huggingface.co/nynxz/MiniMax-H3-PTD-LoRAs) |
| taobao-content-ai ([MiniMax-H3-PTD-LoRAs](https://huggingface.co/taobao-content-ai/MiniMax-H3-PTD-LoRAs)) | Ref2VA | 4, 8 | their bf16 files, or int8 made with this pack |

## Install

It runs on top of the official [ComfyUI-PTD](https://github.com/vita-epfl/PTD-inference) pack. Clone both
into `custom_nodes`; no pip install is needed:

    cd ComfyUI/custom_nodes
    git clone --branch release/comfyui-inference --single-branch https://github.com/vita-epfl/PTD-inference ComfyUI-PTD
    git clone https://github.com/Nynxz/ComfyUI-PTD-Compact

Put the adapter files in `ComfyUI/models/ptd/`, then wire

`Load Diffusion Model → Load PTD MiniMax-H3 Adapters (compact) → Basic Guider + PTD Sampler (compact) +
PTD Sigmas (compact) → SamplerCustomAdvanced`

The step count comes from the file. The official PTD Sampler / Sigmas also work for the EPFL files, but not
for Ref2VA, whose step times are its own. If ComfyUI imports the official pack first, its
`No module named 'ptd'` error in the log is harmless: this pack registers the official nodes itself.

## Options

`kernels`:

- **exact (PTD reference)**, the default: PTD's own bf16 block maths, as the adapters were trained.
- **fast (native kernels)**: ComfyUI's fused native blocks, about 4–8% faster per step. Only the rounding
  order differs.

## Base model

Use the base the adapters were trained on, fl2va for the EPFL files and ref2va for Ref2VA.

| `diffusion_models/` file | Works |
|---|---|
| `minimax_h3_{fl2va,ref2va}_bf16` | yes, the validated bases |
| `minimax_h3_{fl2va,ref2va}_pruned_bf16` | yes, this loader only |
| `…_int8_convrot`, `…_pruned_int8_convrot`, `…_pruned_fp8_scaled` | yes |
| `…_pruned_w6a8` | no, its video head is quantized |

The pruned bases keep every layer PTD adapts byte-identical to the original and replace only the time path
with a small curve table, which matches the original AdaLN to within 0.03% at both releases' step times
(10× below the original's own bf16 rounding). Comfy-Org's ref2va is byte-identical to MiniMax's
`transformer_ref`, which the Ref2VA adapters were trained on.

## int8 files

Each per-step adapter's error against the original is under 1.2% (mean 0.4–0.9%), at a quarter (EPFL) or
half (Ref2VA, from bf16) of the size. To make one from a Ref2VA file:

    python tools/ref2va_int8.py MiniMax-H3-Ref2VA-PTD-4Step.safetensors MiniMax-H3-Ref2VA-PTD-4Step_int8.safetensors

In side-by-side renders (same seed), swapping bf16 for int8 adapters, the pruned int8 base for the pruned
bf16 one, or `exact` for `fast` each changes the video by the same small amount, at the level of rounding
noise. The int8 base runs about 3× faster than the bf16 one on a 32 GB card.

## Notes

- **Memory:** the adapter bank stays in system RAM (int8: about 2.8 GB for a 4-step file). Each layer
  copies only the current step's factors to the GPU.
- **DynamicVRAM:** supported, including ComfyUI's allocation compiler. The log shows per-step residency,
  e.g. `PTD hop 2 VRAM: base model resident 19.5/19.5 GB …`, and a run summary,
  `PTD: 4 hops in 41.8 s (10.45 s/hop)`. If the compiler ever fails (`aimdo memory compile error`), that run
  is sampled again without it and a warning is logged.
- **Ref2VA** output matches taobao-content-ai's own ComfyUI implementation bit for bit in `exact` mode.
- **Other LoRAs** stack with PTD in either order.

## Tests

    PYTHONPATH=/path/to/ComfyUI:/path/to/ComfyUI-PTD:/path/to/custom_nodes \
        python -m pytest tests -q --import-mode=importlib

## License

Apache-2.0, like the official ComfyUI-PTD pack it builds on; the Ref2VA support follows
taobao-content-ai's ComfyUI-MiniMaxH3-PTD (Apache-2.0), see [NOTICE](NOTICE). The adapter weights and
MiniMax-H3 have their own licences; see the model repos.

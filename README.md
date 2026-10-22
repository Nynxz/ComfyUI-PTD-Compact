# ComfyUI-PTD-Compact

One node, **Load PTD MiniMax-H3 Adapters (compact)**, for the compact PTD adapter files in
[nynxz/MiniMax-H3-PTD-LoRAs](https://huggingface.co/nynxz/MiniMax-H3-PTD-LoRAs) (int8 factors, reduced
ranks). It also loads the standard bf16/fp32 PTD files.

## Install

It runs on top of the official [ComfyUI-PTD](https://github.com/vita-epfl/PTD-inference) pack. Clone both
into `custom_nodes`; no pip install is needed:

    cd ComfyUI/custom_nodes
    git clone --branch release/comfyui-inference --single-branch https://github.com/vita-epfl/PTD-inference ComfyUI-PTD
    git clone https://github.com/Nynxz/ComfyUI-PTD-Compact

Put the adapter files in `ComfyUI/models/ptd/`. Use this loader in place of **Load PTD MiniMax-H3 Adapters**
and keep the official **PTD Sampler** and **PTD Sigmas**.

If ComfyUI imports the official pack first, its `No module named 'ptd'` error in the log is harmless:
this pack registers the official nodes itself.

## Options

`kernels`:

- **exact (PTD reference)**, the default: PTD's own bf16 block maths, as the adapters were trained.
- **fast (native kernels)**: ComfyUI's fused native blocks, about 8% faster per block. Only the rounding
  order differs, so compare a seed in both modes before relying on it.

## Base model

PTD was distilled from Comfy-Org's **fl2va** H3. **ref2va** is a different model and is rejected.

| `diffusion_models/` file | Works |
|---|---|
| `minimax_h3_fl2va_bf16` | yes, the validated base |
| `minimax_h3_fl2va_pruned_bf16` | yes, this loader only |
| `minimax_h3_fl2va_int8_convrot`, `…_pruned_int8_convrot`, `…_pruned_fp8_scaled` | yes; quality against bf16 not measured |
| `minimax_h3_fl2va_pruned_w6a8` | no, its video head is quantized |
| `minimax_h3_ref2va_*` | no |

The pruned base keeps every layer PTD adapts byte-identical and replaces only the time path with a small
curve table, which matches the original AdaLN to within 0.03% at PTD's step times.

## Notes

- **Memory:** the adapter bank stays int8 in system RAM (about 2.8 GB for the 4-step file). Each layer
  copies only the current step's factors to the GPU.
- **DynamicVRAM:** supported, including ComfyUI's allocation compiler. The log shows per-step residency,
  e.g. `PTD hop 2 VRAM: base model resident 19.5/19.5 GB …`, and a run summary,
  `PTD: 4 hops in 41.8 s (10.45 s/hop)`. If the compiler ever fails (`aimdo memory compile error`), that run
  is sampled again without it and a warning is logged.
- **Other LoRAs** stack with PTD in either order.

## Tests

    PYTHONPATH=/path/to/ComfyUI:/path/to/ComfyUI-PTD:/path/to/custom_nodes \
        python -m pytest tests -q --import-mode=importlib

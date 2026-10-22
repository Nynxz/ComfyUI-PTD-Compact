"""Write the int8 version of a taobao-content-ai Ref2VA PTD file and report each adapter's error.

    python tools/ref2va_int8.py MiniMax-H3-Ref2VA-PTD-4Step.safetensors ptd_h3_ref2va_4step_int8.safetensors

Needs torch and safetensors only. The error of each hop's B.A against the original is exact, computed from
rank x rank Gram matrices.
"""
import importlib.util
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

spec = importlib.util.spec_from_file_location("pdd", os.path.join(os.path.dirname(__file__), "..", "pdd.py"))
pdd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pdd)

src, dst = sys.argv[1], sys.argv[2]
with safe_open(src, "pt") as f:
    meta = dict(f.metadata() or {})
    tensors = {k: f.get_tensor(k) for k in f.keys()}
if not pdd.is_pdd(meta):
    sys.exit(f"{src} is not a Ref2VA PTD file (no ptd_num_hops metadata)")
q, qmeta = pdd.to_int8(tensors, meta)
qmeta["ptd_modified"] = json.dumps({
    "note": "Modified from taobao-content-ai/MiniMax-H3-PTD-LoRAs: lora_down / lora_up quantized to int8 with one "
            "float32 scale per row (github.com/Nynxz/ComfyUI-PTD-Compact). PDD heads unchanged.",
    "source": os.path.basename(src)})

errors = []
for key in tensors:
    if key.endswith(".lora_down"):
        module = key[:-len(".lora_down")]
        a, b = tensors[key].float(), tensors[module + ".lora_up"].float()
        qa = q[key].float() * q[key + "_scale"][..., None]
        qb = q[module + ".lora_up"].float() * q[module + ".lora_up_scale"][..., None]
        for h in range(a.shape[0]):
            left, right = torch.cat([b[h], -qb[h]], 1), torch.cat([a[h], qa[h]], 0)
            err = torch.trace((left.T @ left) @ (right @ right.T)).clamp_min(0)
            ref = torch.trace((b[h].T @ b[h]) @ (a[h] @ a[h].T))
            errors.append(float((err / ref).sqrt()))
errors = torch.tensor(errors)
save_file(q, dst, metadata=qmeta)
print(f"{len(errors)} adapters, relative error mean {errors.mean() * 100:.2f}% max {errors.max() * 100:.2f}%; "
      f"{os.path.getsize(dst) / 1e9:.2f} GB written to {dst}")

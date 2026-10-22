"""Both packs cloned into custom_nodes, nothing pip-installed: ComfyUI loads them in either order and the
compact loader's MODEL works with the PTD Sampler/Sigmas that end up registered. Each order runs in a
fresh interpreter whose sys.path has ComfyUI but not the PTD checkout.

Needs COMFYUI (a ComfyUI checkout) and PTD_REPO (an official PTD-inference checkout) in the environment.
"""
import os
import subprocess
import sys
import textwrap

import pytest

COMFYUI, PTD_REPO = os.environ.get("COMFYUI"), os.environ.get("PTD_REPO")
PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytestmark = pytest.mark.skipif(not (COMFYUI and PTD_REPO), reason="set COMFYUI and PTD_REPO")

SCRIPT = textwrap.dedent("""
    import asyncio, importlib.util, os, sys
    import comfy.cli_args; comfy.cli_args.args.cpu = True
    import nodes, folder_paths
    assert importlib.util.find_spec("ptd") is None, "ptd must not be importable up front"
    root = sys.argv[1]
    for name in sys.argv[2:]:
        asyncio.run(nodes.load_custom_node(os.path.join(root, name)))
    for node in ("PTDH3Loader", "PTDH3Sampler", "PTDH3Sigmas", "PTDH3CompactLoader"):
        assert node in nodes.NODE_CLASS_MAPPINGS, node
    sys.path.insert(0, os.path.join(root, sys.argv[-1] if "Compact" in sys.argv[-1] else sys.argv[-2], "tests"))
    import test_stacking as t
    base = t.native_model()
    t.adapter_file(os.path.join(root, "a.safetensors"), base.get_model_object("diffusion_model"))
    folder_paths.add_model_folder_path("ptd", root)
    model, = nodes.NODE_CLASS_MAPPINGS["PTDH3CompactLoader"]().load(base, "a.safetensors")
    nodes.NODE_CLASS_MAPPINGS["PTDH3Sampler"]().get_sampler(model)
    sigmas, = nodes.NODE_CLASS_MAPPINGS["PTDH3Sigmas"]().get_sigmas(model)
    assert sigmas.tolist() == [0.999, 0.947, 0.857, 0.667, 0.0]
    print("OK", nodes.NODE_CLASS_MAPPINGS["PTDH3Sampler"].__module__)
""")


@pytest.mark.parametrize("order", [("ComfyUI-PTD", "ComfyUI-PTD-Compact"), ("ComfyUI-PTD-Compact", "ComfyUI-PTD")])
def test_clone_only_install(tmp_path, order):
    os.symlink(PTD_REPO, tmp_path / "ComfyUI-PTD")
    os.symlink(PACK, tmp_path / "ComfyUI-PTD-Compact")
    env = dict(os.environ, PYTHONPATH=COMFYUI)
    run = subprocess.run([sys.executable, "-c", SCRIPT, str(tmp_path), *order], cwd=COMFYUI, env=env,
                         capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    assert "OK" in run.stdout

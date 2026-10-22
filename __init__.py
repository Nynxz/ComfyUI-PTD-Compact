import importlib.util
import os
import sys

# The official ComfyUI-PTD code imports its own packages absolutely (``from ptd.recipes import ...``),
# which its README handles with ``pip install -e``. Instead, find its checkout next to this pack and put
# it on the import path, so a plain ``git clone`` into custom_nodes is enough.
_custom_nodes = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if importlib.util.find_spec("ptd") is None:
    for _name in sorted(os.listdir(_custom_nodes)):
        _path = os.path.join(_custom_nodes, _name)
        if os.path.isfile(os.path.join(_path, "ptd", "recipes.py")) and \
                os.path.isfile(os.path.join(_path, "ptd_comfy", "patcher.py")):
            sys.path.append(_path)
            break

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

# If the official pack was imported before the path fix, it failed to register. Register its nodes
# from here; if it loads after this pack instead, it registers itself and replaces these.
try:
    import nodes as _comfy_nodes
    if "PTDH3Sampler" not in _comfy_nodes.NODE_CLASS_MAPPINGS and importlib.util.find_spec("ptd_comfy"):
        from ptd_comfy import nodes as _official
        NODE_CLASS_MAPPINGS = {**_official.NODE_CLASS_MAPPINGS, **NODE_CLASS_MAPPINGS}
        NODE_DISPLAY_NAME_MAPPINGS = {**getattr(_official, "NODE_DISPLAY_NAME_MAPPINGS", {}), **NODE_DISPLAY_NAME_MAPPINGS}
except ImportError:  # outside ComfyUI
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

"""LTX Video node package: latent sizing and the two-stage distilled samplers.

Live previews during sampling (including the taeltx decoder LTX 2.3 needs, since
core ships none) moved out to the standalone Live-Preview-MXD pack.
"""
import importlib

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for _name in (
    "latents",
    "samplers",
):
    try:
        _mod = importlib.import_module(f".{_name}", __name__)
    except Exception as e:
        print(f"[ComfyUI-MaxedOut] Failed to import 'nodes.ltx.{_name}': {e}")
        continue
    NODE_CLASS_MAPPINGS.update(getattr(_mod, "NODE_CLASS_MAPPINGS", {}) or {})
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_mod, "NODE_DISPLAY_NAME_MAPPINGS", {}) or {})

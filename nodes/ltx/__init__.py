"""LTX Video node package: latent sizing and the two-stage distilled samplers.

The optional automatic taeltx preview support lives in preview.py and is loaded
by system.live_preview only when its backend hook needs the LTX 2.3 fallback.
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

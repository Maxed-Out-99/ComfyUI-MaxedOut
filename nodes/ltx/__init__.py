"""LTX Video node package: latent sizing, two-stage samplers, taeltx live preview."""
import importlib

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for _name in (
    "latents",
    "samplers",
    "preview",
):
    try:
        _mod = importlib.import_module(f".{_name}", __name__)
    except Exception as e:
        print(f"[ComfyUI-MaxedOut] Failed to import 'nodes.ltx.{_name}': {e}")
        continue
    NODE_CLASS_MAPPINGS.update(getattr(_mod, "NODE_CLASS_MAPPINGS", {}) or {})
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_mod, "NODE_DISPLAY_NAME_MAPPINGS", {}) or {})

"""WAN 2.2 node package: buckets/scalers, latent save-load, I2V conditioning, video ops."""
import importlib

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for _name in (
    "buckets",
    "latent_io",
    "i2v",
    "video_ops",
):
    try:
        _mod = importlib.import_module(f".{_name}", __name__)
    except Exception as e:
        print(f"[ComfyUI-MaxedOut] Failed to import 'nodes.wan22.{_name}': {e}")
        continue
    NODE_CLASS_MAPPINGS.update(getattr(_mod, "NODE_CLASS_MAPPINGS", {}) or {})
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_mod, "NODE_DISPLAY_NAME_MAPPINGS", {}) or {})

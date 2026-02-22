import importlib

WEB_DIRECTORY = "web"

def _safe_import(module_name: str):
    try:
        return importlib.import_module(f".{module_name}", __name__)
    except Exception as e:
        print(f"[ComfyUI-MaxedOut] Failed to import '{module_name}': {e}")
        return None

def _get_mappings(mod):
    if mod is None:
        return {}, {}
    class_map = getattr(mod, "NODE_CLASS_MAPPINGS", {}) or {}
    display_map = getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", {}) or {}
    return class_map, display_map

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for _name in ("maxedoutnodes", "mediacomparers", "wan22nodes", "loraloader_mxd", "wan_svi_first_last_mxd"):
    _mod = _safe_import(_name)
    _class_map, _display_map = _get_mappings(_mod)
    NODE_CLASS_MAPPINGS.update(_class_map)
    NODE_DISPLAY_NAME_MAPPINGS.update(_display_map)

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]


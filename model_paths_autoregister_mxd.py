r"""Auto-register every subfolder of a user-configured model storage root
with folder_paths, the same way ComfyUI/models/<type> works.

The root is resolved in this order (first hit wins):
  1. MAXEDOUT_MODEL_STORAGE environment variable
  2. model_storage_config.json next to this file (gitignored -- copy
     model_storage_config.json.example to create your own, it never gets
     committed)
  3. The "MXD > Model Storage > Root Folder" setting in the ComfyUI
     settings panel (web/model_storage_settings_mxd.js). Takes effect on
     the next server restart since folder registration happens at import
     time.

If none of these are set, nothing is registered and ComfyUI behaves as
usual -- this is entirely opt-in.
"""

import json
import os

try:
    import folder_paths
except ImportError:
    folder_paths = None

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_THIS_DIR, "model_storage_config.json")


def _root_from_env():
    return os.environ.get("MAXEDOUT_MODEL_STORAGE") or None


def _root_from_config_file():
    if not os.path.isfile(_CONFIG_PATH):
        return None
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("model_storage_root") or None
    except Exception as e:
        print(f"[ComfyUI-MaxedOut] Failed to read {_CONFIG_PATH}: {e}")
        return None


def _root_from_comfyui_settings():
    if folder_paths is None:
        return None
    try:
        settings_path = os.path.join(folder_paths.get_user_directory(), "default", "comfy.settings.json")
        if not os.path.isfile(settings_path):
            return None
        with open(settings_path, "r", encoding="utf-8") as f:
            settings = json.load(f)
        return settings.get("MXD.ModelStorageRoot") or None
    except Exception:
        return None


def _resolve_model_storage_root():
    for source in (_root_from_env, _root_from_config_file, _root_from_comfyui_settings):
        root = source()
        if root:
            return root
    return None


try:
    _root = _resolve_model_storage_root()
    if folder_paths is not None and _root and os.path.isdir(_root):
        count = 0
        for item_name in os.listdir(_root):
            item_path = os.path.join(_root, item_name)
            if os.path.isdir(item_path):
                folder_paths.add_model_folder_path(item_name, item_path)
                count += 1

        if count > 0:
            print(f"[ComfyUI-MaxedOut] Auto-registered {count} model folders from {_root}")
    elif _root and not os.path.isdir(_root):
        print(f"[ComfyUI-MaxedOut] Model storage root '{_root}' is not a valid directory, skipping auto-register")
except Exception as e:
    print(f"[ComfyUI-MaxedOut] Failed to auto-register model folders: {e}")


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

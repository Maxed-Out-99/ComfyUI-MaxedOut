"""Dump every registered node's public schema to JSON, for refactor-safety diffing.

This is the repo's "nothing broke" gate: it imports the pack exactly the way
ComfyUI does (spec_from_file_location against __init__.py) with the real
installed core on sys.path, then records each node's key, display name,
category, FUNCTION, OUTPUT_NODE, INPUT_TYPES, RETURN_TYPES and RETURN_NAMES.

Usage (run with the desktop venv python):
    python scripts/dump_node_schema.py out.json
    python scripts/dump_node_schema.py --compare baseline.json current.json

Environment:
    COMFYUI_ROOT  path to the installed ComfyUI core
                  (default: C:\\Users\\user\\ComfyUI-Installs\\ComfyUI\\ComfyUI)

Dynamic combo inputs (file/folder lists from folder_paths or disk scans) are
normalized to presence-only so the dump is stable while the user's model and
output folders keep changing. Static enum combos (short, no dots/slashes in
entries) are recorded verbatim.
"""

import json
import os
import re
import sys
import types

DEFAULT_COMFY = r"C:\Users\user\ComfyUI-Installs\ComfyUI\ComfyUI"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACK_NAME = "ComfyUI-MaxedOut"

# Combos with more entries than this, or with path-like entries, are treated
# as dynamic file lists and compared by presence only.
STATIC_COMBO_MAX = 30
_PATHY = re.compile(r"[./\\]")

# Input names whose combo entries come from disk scans or user data (output
# subfolders, saved characters, video files) — always presence-only, even when
# the entries happen to look like a static enum.
DYNAMIC_INPUT_NAMES = {"folder", "subfolder", "selected_character", "video"}


def _norm_combo(values, force_dynamic=False):
    vals = [str(v) for v in values]
    if (
        not force_dynamic
        and vals
        and len(vals) <= STATIC_COMBO_MAX
        and not any(_PATHY.search(v) for v in vals)
    ):
        return {"combo": vals}
    return {"combo": "nonempty" if vals else "empty", "dynamic": True}


def _norm(obj, in_type_position=False):
    if isinstance(obj, (list, tuple)):
        if in_type_position:
            return _norm_combo(obj)
        return [_norm(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _norm(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def _norm_input_spec(spec, name=""):
    """spec is (type,) or (type, config) or just type."""
    force_dynamic = name in DYNAMIC_INPUT_NAMES

    def norm_type(t):
        if isinstance(t, (list, tuple)):
            return _norm_combo(t, force_dynamic)
        return _norm(t)

    if not isinstance(spec, (list, tuple)):
        return {"type": norm_type(spec)}
    out = {"type": norm_type(spec[0])}
    if len(spec) > 1 and isinstance(spec[1], dict):
        cfg = dict(spec[1])
        # A default drawn from a dynamic file list is as unstable as the list.
        if isinstance(out["type"], dict) and out["type"].get("dynamic"):
            cfg.pop("default", None)
            cfg["default"] = "<dynamic-combo-default>"
        out["config"] = _norm(cfg)
    elif len(spec) > 1:
        out["extra"] = _norm(list(spec[1:]))
    return out


def _norm_input_types(it):
    out = {}
    for section in ("required", "optional", "hidden"):
        sec = (it or {}).get(section)
        if not isinstance(sec, dict):
            continue
        out[section] = {name: _norm_input_spec(spec, name) for name, spec in sorted(sec.items())}
    return out


def _stub_module(name):
    class _Dummy:
        def __call__(self, *a, **k):
            return None

        def __getattr__(self, _):
            return _Dummy()

        def __bool__(self):
            return False

    mod = types.ModuleType(name)
    mod.__getattr__ = lambda attr: _Dummy()  # type: ignore[attr-defined]
    sys.modules[name] = mod
    print(f"[dump_node_schema] stubbed missing module: {name}")


def _import_pack(comfy_root):
    sys.path.insert(0, comfy_root)
    # comfy.cli_args parses sys.argv on import; keep our args out of it.
    sys.argv = [sys.argv[0]]

    # Core's nodes.py prepends <core>/comfy to sys.path, after which a bare
    # `import utils` finds comfy/utils.py instead of the core utils package.
    # Real ComfyUI dodges this because main.py imports utils.* first; do the same.
    try:
        import utils.install_util  # noqa: F401
    except ImportError:
        pass

    import importlib.util

    for _ in range(10):
        try:
            import asyncio

            import server  # noqa: F401  (ComfyUI's server module)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            if getattr(server.PromptServer, "instance", None) is None:
                server.PromptServer(loop)
            break
        except ModuleNotFoundError as e:
            _stub_module(e.name)
    else:
        raise RuntimeError("could not import ComfyUI server after stubbing 10 modules")

    init_py = os.path.join(REPO_ROOT, "__init__.py")
    spec = importlib.util.spec_from_file_location(PACK_NAME, init_py)
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACK_NAME] = module
    spec.loader.exec_module(module)
    return module


def dump(comfy_root, out_path):
    pack = _import_pack(comfy_root)
    class_map = pack.NODE_CLASS_MAPPINGS
    display_map = pack.NODE_DISPLAY_NAME_MAPPINGS

    result = {"node_count": len(class_map), "nodes": {}}
    for key in sorted(class_map):
        cls = class_map[key]
        entry = {
            "class": cls.__name__,
            "display_name": display_map.get(key),
            "category": getattr(cls, "CATEGORY", None),
            "function": getattr(cls, "FUNCTION", None),
            "output_node": bool(getattr(cls, "OUTPUT_NODE", False)),
        }
        try:
            entry["input_types"] = _norm_input_types(cls.INPUT_TYPES())
        except Exception as e:  # record, never crash the dump
            entry["input_types"] = {"__error__": f"{type(e).__name__}: {e}"}
        rt = getattr(cls, "RETURN_TYPES", None)
        entry["return_types"] = [str(t) for t in rt] if rt is not None else None
        rn = getattr(cls, "RETURN_NAMES", None)
        entry["return_names"] = [str(n) for n in rn] if rn is not None else None
        result["nodes"][key] = entry

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    print(f"[dump_node_schema] wrote {len(class_map)} nodes to {out_path}")
    errors = [k for k, v in result["nodes"].items() if "__error__" in v.get("input_types", {})]
    if errors:
        print(f"[dump_node_schema] WARNING: INPUT_TYPES failed for: {errors}")
    return 0


def compare(a_path, b_path):
    with open(a_path, encoding="utf-8") as f:
        a = json.load(f)
    with open(b_path, encoding="utf-8") as f:
        b = json.load(f)
    ok = True
    a_nodes, b_nodes = a["nodes"], b["nodes"]
    for key in sorted(set(a_nodes) | set(b_nodes)):
        if key not in b_nodes:
            print(f"MISSING node: {key}")
            ok = False
        elif key not in a_nodes:
            print(f"NEW node: {key}")
            ok = False
        elif a_nodes[key] != b_nodes[key]:
            print(f"CHANGED node: {key}")
            for field in sorted(set(a_nodes[key]) | set(b_nodes[key])):
                av, bv = a_nodes[key].get(field), b_nodes[key].get(field)
                if av != bv:
                    print(f"  {field}:")
                    print(f"    baseline: {json.dumps(av, sort_keys=True)[:400]}")
                    print(f"    current:  {json.dumps(bv, sort_keys=True)[:400]}")
            ok = False
    if ok:
        print(f"OK: {len(a_nodes)} nodes identical")
    return 0 if ok else 1


def main():
    args = sys.argv[1:]
    if args and args[0] == "--compare":
        sys.exit(compare(args[1], args[2]))
    out = args[0] if args else os.path.join(REPO_ROOT, "scripts", "node_schema_current.json")
    comfy_root = os.environ.get("COMFYUI_ROOT", DEFAULT_COMFY)
    sys.exit(dump(comfy_root, out))


if __name__ == "__main__":
    main()

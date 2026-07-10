"""WAN 2.2 latent save/load: two-stage I2V handoff via .latent + .cond.pt sidecar files.

Registered nodes:
  SaveLatent_I2V_MXD                   Save Latent MXD
  LoadLatent_I2V_MXD                   Load Latent MXD
  LoadLatents_FromFolder_I2V_MXD       Load Latent Batch MXD
  LoadLatent_I2V_Pipe_MXD              Load Latent Pipe MXD
  LoadLatents_FromFolder_I2V_Pipe_MXD  Load Latent Batch Pipe MXD
  LatentPipeUnpack_MXD                 Unpack Latent Pipe MXD

Route: GET /mxd/latents/files (fresh re-scan for the run_folder queuing loop).

Latents are stored under <input>/latents. Each .latent embeds the source
workflow + KSampler params in safetensors metadata; conditioning goes in a
.cond.pt sidecar. The loaders reconstruct sampler settings from that metadata
so the second stage can resume with matching parameters.
"""
from __future__ import annotations
import os, re, glob, json, hashlib, copy
from collections import deque
from typing import Any, Dict, Tuple, Optional, List, Union

import torch
from safetensors import safe_open

import folder_paths
import comfy.utils
from comfy.cli_args import args
from nodes import KSamplerAdvanced

from aiohttp import web

from ..shared.metadata import _safe_json_loads
from ..shared.paths import _sort_paths_newest_first, _strip_counter
from ..shared.routes import register_get_route


def _sort_latent_options_by_folder(options: List[str], root: str = "") -> List[str]:
    """
    Order relative '.latent' option paths so that the combo's prev/next arrows
    stay confined to one folder before moving on, newest first:
        folderA/file1, folderA/file2, ..., folderB/file1, ...
    Folders are ordered by the mtime of their most recently modified file (so
    a folder that just received a new file jumps back to the top), and files
    within each folder are newest first.
    """
    def _mtime(rel: str) -> float:
        if not root:
            return 0.0
        try:
            return os.path.getmtime(os.path.join(root, rel))
        except OSError:
            return 0.0

    folder_of = lambda rel: rel.replace("\\", "/").rsplit("/", 1)[0] if "/" in rel.replace("\\", "/") else ""

    folder_latest: Dict[str, float] = {}
    for rel in options:
        folder = folder_of(rel)
        m = _mtime(rel)
        if m > folder_latest.get(folder, -1.0):
            folder_latest[folder] = m

    def key(rel: str):
        folder = folder_of(rel)
        return (-folder_latest.get(folder, 0.0), -_mtime(rel))

    return sorted(options, key=key)

def _list_latent_subfolders(latents_root: str) -> List[str]:
    """
    List latent subfolders recursively (e.g. "a", "a/b"), newest first by
    latest latent mtime in each branch.
    """
    files = glob.glob(os.path.join(latents_root, "**", "*.latent"), recursive=True)
    if not files:
        return []

    folder_latest_mtime: Dict[str, float] = {}
    for file_path in files:
        rel_dir = os.path.relpath(os.path.dirname(file_path), latents_root).replace(os.sep, "/").strip("/")
        if not rel_dir or rel_dir == ".":
            continue
        try:
            mtime = os.path.getmtime(file_path)
        except OSError:
            mtime = 0.0

        # Include each ancestor so both "a" and "a/b" appear as options.
        parts = [p for p in rel_dir.split("/") if p]
        for i in range(1, len(parts) + 1):
            branch = "/".join(parts[:i])
            prev = folder_latest_mtime.get(branch, -1.0)
            if mtime > prev:
                folder_latest_mtime[branch] = mtime

    return [
        folder
        for folder, _ in sorted(
            folder_latest_mtime.items(),
            key=lambda kv: (-kv[1], kv[0].lower()),
        )
    ]


async def mxd_list_latent_files(request):
    """
    Fresh re-scan of input/latents for .latent files. Used by the run_folder
    queuing loop (refresh_before_run) to pick up files a still-running
    workflow is writing concurrently, instead of relying on the dropdown
    list captured whenever the node's combo was last populated.
    """
    latents_root = os.path.join(folder_paths.get_input_directory(), "latents")
    os.makedirs(latents_root, exist_ok=True)
    files = glob.glob(os.path.join(latents_root, "**", "*.latent"), recursive=True)
    options = [os.path.relpath(f, latents_root).replace(os.sep, "/") for f in files]
    options = _sort_latent_options_by_folder(options, latents_root)
    return web.json_response(options)


register_get_route("/mxd/latents/files", mxd_list_latent_files)


# ---------- SaveLatent (saves latent + conditioning + optional trim_latent) ----------
class SaveLatent_I2V_MXD:
    """
    Default latent saver, works for t2v, i2v, and VACE 2.2. Persists:
      • latent tensor  ->  .latent
      • pos/neg CONDITIONING  ->  .cond.pt
      • optional trim_latent value (VACE 2.2)  ->  .cond.pt
    """
    TITLE = "Save Latent MXD"
    CATEGORY = "MXD/Latents"
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "save_only"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "Latent to save."}),
                "positive": ("CONDITIONING", {"tooltip": "Positive CONDITIONING to save alongside the latent."}),
                "negative": ("CONDITIONING", {"tooltip": "Negative CONDITIONING to save alongside the latent."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "Prefix for saved files"}),
            },
            "optional": {
                "trim_latent": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "VACE 2.2 trim_latent value to preserve with this latent. Usually 0 or 1. Ignored for t2v/i2v."
                }),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"},
        }

    def save_only(self, samples, positive, negative, filename_prefix="ComfyUI",
                  trim_latent=0, prompt=None, extra_pnginfo=None, unique_id=None):
        _save_i2v_latent_bundle(
            samples=samples,
            positive=positive,
            negative=negative,
            filename_prefix=filename_prefix,
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
            unique_id=unique_id,
            sidecar_extra={"trim_latent": _coerce_trim_latent(trim_latent)},
        )
        return {}

# ---------- Helpers ----------
def _load_latent_file(latent_path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], List[str]]:
    """
    Load safetensors latent with Comfy metadata.
    Returns (samples_dict, metadata_dict, keys_list)
    """
    with safe_open(latent_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        # prefer explicit key we write
        if "latent_tensor" in keys:
            t = f.get_tensor("latent_tensor").float().contiguous()
        else:
            # fall back (some variants might save using a different name)
            first = keys[0]
            t = f.get_tensor(first).float().contiguous()

        meta = f.metadata() or {}

        # if ancient format, rescale (match Comfy behavior)
        if "latent_format_version_0" not in keys:
            t = t * (1.0 / 0.18215)

    return {"samples": t}, meta, keys


def _workflow_node_bbox(nodes: List[Dict[str, Any]]) -> Optional[Tuple[float, float, float, float]]:
    """(min_x, min_y, max_x, max_y) over a litegraph 'nodes' list. None if no positions found."""
    xs0, ys0, xs1, ys1 = [], [], [], []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        pos = n.get("pos")
        if isinstance(pos, list) and len(pos) >= 2:
            x, y = pos[0], pos[1]
        elif isinstance(pos, dict):
            x, y = pos.get("0", 0), pos.get("1", 0)
        else:
            continue
        size = n.get("size")
        w = size[0] if isinstance(size, (list, tuple)) and len(size) >= 1 else 200
        h = size[1] if isinstance(size, (list, tuple)) and len(size) >= 2 else 100
        xs0.append(x); ys0.append(y); xs1.append(x + w); ys1.append(y + h)
    if not xs0:
        return None
    return (min(xs0), min(ys0), max(xs1), max(ys1))


def _offset_workflow_nodes(nodes: List[Dict[str, Any]], dx: float, dy: float) -> None:
    for n in nodes:
        if not isinstance(n, dict):
            continue
        pos = n.get("pos")
        if isinstance(pos, list) and len(pos) >= 2:
            pos[0] = pos[0] + dx
            pos[1] = pos[1] + dy
        elif isinstance(pos, dict):
            if "0" in pos: pos["0"] = pos["0"] + dx
            if "1" in pos: pos["1"] = pos["1"] + dy


def _merge_prior_workflow_into_current(prior_workflow_json: Optional[str], current_workflow: Any) -> Any:
    """
    Merge a previously-saved workflow graph (embedded in a loaded .latent file) into the
    workflow graph of the run that's currently saving. The prior graph's nodes/links/groups
    are copied in with fresh ids and shifted to sit to the left of the current graph, wrapped
    in a labelled group - so dragging the final video into ComfyUI shows both stages at once,
    the same as if you'd copy/pasted the first workflow onto the second one's canvas.

    Best-effort: on any parse/shape problem, returns current_workflow untouched.
    """
    if not prior_workflow_json or not isinstance(current_workflow, dict):
        return current_workflow

    try:
        prior = json.loads(prior_workflow_json) if isinstance(prior_workflow_json, str) else prior_workflow_json
        if not isinstance(prior, dict):
            return current_workflow

        prior_nodes = prior.get("nodes")
        if not isinstance(prior_nodes, list) or not prior_nodes:
            return current_workflow

        merged = copy.deepcopy(current_workflow)
        current_nodes = merged.get("nodes")
        if not isinstance(current_nodes, list):
            current_nodes = []
            merged["nodes"] = current_nodes

        prior_nodes = copy.deepcopy(prior_nodes)
        prior_links = copy.deepcopy(prior.get("links")) if isinstance(prior.get("links"), list) else []
        prior_groups = copy.deepcopy(prior.get("groups")) if isinstance(prior.get("groups"), list) else []

        # ---- remap node ids so they can't collide with the current graph ----
        current_last_node_id = merged.get("last_node_id")
        if not isinstance(current_last_node_id, int):
            current_last_node_id = max((n.get("id", 0) for n in current_nodes if isinstance(n, dict)), default=0)
        next_node_id = current_last_node_id + 1
        node_id_map: Dict[Any, int] = {}
        for n in prior_nodes:
            if not isinstance(n, dict) or "id" not in n:
                continue
            node_id_map[n["id"]] = next_node_id
            n["id"] = next_node_id
            next_node_id += 1

        # ---- remap link ids the same way ----
        current_last_link_id = merged.get("last_link_id")
        if not isinstance(current_last_link_id, int):
            current_last_link_id = max(
                (l[0] for l in (merged.get("links") or []) if isinstance(l, list) and l), default=0
            )
        next_link_id = current_last_link_id + 1
        link_id_map: Dict[Any, int] = {}
        for l in prior_links:
            if isinstance(l, list) and l:
                link_id_map[l[0]] = next_link_id
                next_link_id += 1

        for n in prior_nodes:
            if not isinstance(n, dict):
                continue
            for inp in (n.get("inputs") or []):
                if isinstance(inp, dict) and inp.get("link") is not None:
                    inp["link"] = link_id_map.get(inp["link"], inp["link"])
            for out in (n.get("outputs") or []):
                if isinstance(out, dict) and isinstance(out.get("links"), list):
                    out["links"] = [link_id_map.get(x, x) for x in out["links"]]

        remapped_links = []
        for l in prior_links:
            if not isinstance(l, list) or len(l) < 5:
                continue
            new_l = list(l)
            new_l[0] = link_id_map.get(l[0], l[0])
            new_l[1] = node_id_map.get(l[1], l[1])
            new_l[3] = node_id_map.get(l[3], l[3])
            remapped_links.append(new_l)

        # ---- shift the prior graph so it sits to the left of the current one ----
        current_bbox = _workflow_node_bbox(current_nodes)
        prior_bbox = _workflow_node_bbox(prior_nodes)
        margin = 400
        if current_bbox and prior_bbox:
            dx = (current_bbox[0] - margin) - prior_bbox[2]
            dy = current_bbox[1] - prior_bbox[1]
        else:
            dx, dy = 0, 0
        _offset_workflow_nodes(prior_nodes, dx, dy)
        for g in prior_groups:
            if not isinstance(g, dict):
                continue
            b = g.get("bounding")
            if isinstance(b, list) and len(b) >= 2:
                b[0] = b[0] + dx
                b[1] = b[1] + dy

        # wrap the prior graph in a labelled group so it's obvious what it is
        wrapper_group = None
        prior_bbox_shifted = _workflow_node_bbox(prior_nodes)
        if prior_bbox_shifted:
            pad = 60
            wrapper_group = {
                "title": "Prior stage (loaded latent's source workflow)",
                "bounding": [
                    prior_bbox_shifted[0] - pad,
                    prior_bbox_shifted[1] - pad - 40,
                    (prior_bbox_shifted[2] - prior_bbox_shifted[0]) + pad * 2,
                    (prior_bbox_shifted[3] - prior_bbox_shifted[1]) + pad * 2 + 40,
                ],
                "color": "#3f789e",
                "font_size": 24,
            }

        merged["nodes"] = current_nodes + prior_nodes
        merged["links"] = (merged.get("links") or []) + remapped_links
        groups = list(merged.get("groups") or []) + prior_groups
        if wrapper_group:
            groups.append(wrapper_group)
        merged["groups"] = groups
        merged["last_node_id"] = next_node_id - 1
        merged["last_link_id"] = next_link_id - 1
        return merged
    except Exception as e:
        print(f"[SaveVideoMXD] Could not merge prior stage workflow into embedded metadata: {e}")
        return current_workflow


def _node_sort_key(node_id: str) -> Tuple[int, Union[int, str]]:
    s = str(node_id)
    try:
        return (0, int(s))
    except Exception:
        return (1, s)


def _normalize_prompt_graph(prompt_json: Any) -> Dict[str, Any]:
    if not isinstance(prompt_json, dict):
        return {}
    graph = prompt_json.get("prompt", prompt_json)
    return graph if isinstance(graph, dict) else {}


def _get_graph_node(graph: Dict[str, Any], node_id: Any) -> Optional[Dict[str, Any]]:
    if node_id is None or not isinstance(graph, dict):
        return None
    node = graph.get(str(node_id))
    return node if isinstance(node, dict) else None


def _iter_graph_nodes_sorted(graph: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    nodes: List[Tuple[str, Dict[str, Any]]] = []
    for node_id, node in graph.items():
        if isinstance(node, dict):
            nodes.append((str(node_id), node))
    nodes.sort(key=lambda pair: _node_sort_key(pair[0]))
    return nodes


def _linked_node_id(value: Any) -> Optional[str]:
    if isinstance(value, (list, tuple)) and len(value) >= 1:
        return str(value[0])
    return None


def _is_ksampler_node(node: Any) -> bool:
    if not isinstance(node, dict):
        return False
    return "KSampler" in str(node.get("class_type", ""))


def _collect_upstream_linked_node_ids(node: Dict[str, Any]) -> List[str]:
    inputs = node.get("inputs", {})
    if not isinstance(inputs, dict):
        return []

    seen = set()
    ordered = []

    # Prefer latent-carrying links first.
    for key in ("samples", "latent", "latent_image"):
        linked = _linked_node_id(inputs.get(key))
        if linked is not None and linked not in seen:
            seen.add(linked)
            ordered.append(linked)

    # Then search all other connected inputs in stable order.
    for key, value in inputs.items():
        if key in ("samples", "latent", "latent_image"):
            continue
        linked = _linked_node_id(value)
        if linked is not None and linked not in seen:
            seen.add(linked)
            ordered.append(linked)

    return ordered


def _find_upstream_ksampler_node_id(graph: Dict[str, Any], start_node_id: Any) -> Optional[str]:
    if not isinstance(graph, dict) or start_node_id is None:
        return None

    queue: deque[str] = deque([str(start_node_id)])
    visited = set()

    while queue:
        node_id = queue.popleft()
        if node_id in visited:
            continue
        visited.add(node_id)

        node = _get_graph_node(graph, node_id)
        if not node:
            continue
        if _is_ksampler_node(node):
            return node_id

        for upstream_id in _collect_upstream_linked_node_ids(node):
            if upstream_id not in visited:
                queue.append(upstream_id)

    return None


def _extract_ksampler_params(node: Dict[str, Any]) -> Dict[str, Any]:
    inputs = node.get("inputs", {}) if isinstance(node, dict) else {}
    if not isinstance(inputs, dict):
        inputs = {}

    out: Dict[str, Any] = {}

    def set_int(key: str):
        if key in inputs:
            try:
                out[key] = int(inputs[key])
            except Exception:
                pass

    def set_float(key: str):
        if key in inputs:
            try:
                out[key] = float(inputs[key])
            except Exception:
                pass

    def set_str(key: str):
        if key in inputs and not isinstance(inputs[key], (list, tuple, dict)):
            try:
                out[key] = str(inputs[key]).strip()
            except Exception:
                pass

    set_int("steps")
    set_float("cfg")
    set_str("sampler_name")
    set_str("scheduler")
    set_int("start_at_step")
    set_int("end_at_step")

    return out


def _attach_source_ksampler_metadata(meta: Dict[str, Any], prompt: Any, unique_id: Any) -> None:
    if not isinstance(meta, dict):
        return

    graph = _normalize_prompt_graph(prompt)
    if not graph:
        return

    save_node_id = str(unique_id) if unique_id is not None else ""
    if not save_node_id:
        return

    save_node = _get_graph_node(graph, save_node_id)
    if not save_node:
        return

    source_candidates = _collect_upstream_linked_node_ids(save_node)
    if not source_candidates:
        return

    source_ksampler_id = None
    for start_id in source_candidates:
        source_ksampler_id = _find_upstream_ksampler_node_id(graph, start_id)
        if source_ksampler_id:
            break

    if not source_ksampler_id:
        return

    source_node = _get_graph_node(graph, source_ksampler_id)
    if not source_node:
        return

    meta["mxd_source_save_node_id"] = save_node_id
    meta["mxd_source_ksampler_node_id"] = source_ksampler_id
    try:
        meta["mxd_source_ksampler_params"] = json.dumps(_extract_ksampler_params(source_node))
    except Exception:
        pass


def _build_latent_metadata(prompt=None, extra_pnginfo=None, unique_id=None, extra_meta=None):
    if args.disable_metadata:
        return None

    meta = {}
    if prompt is not None:
        try:
            meta["prompt"] = json.dumps(prompt)
        except Exception:
            pass
    if extra_pnginfo is not None:
        for k, v in extra_pnginfo.items():
            try:
                meta[k] = json.dumps(v)
            except Exception:
                pass
    if isinstance(extra_meta, dict):
        for k, v in extra_meta.items():
            try:
                meta[str(k)] = json.dumps(v)
            except Exception:
                pass
    _attach_source_ksampler_metadata(meta, prompt, unique_id)
    return meta


def _save_i2v_latent_bundle(
    samples,
    positive,
    negative,
    filename_prefix="I2V",
    prompt=None,
    extra_pnginfo=None,
    unique_id=None,
    sidecar_extra=None,
):
    latents_dir = os.path.join(folder_paths.get_input_directory(), "latents")
    os.makedirs(latents_dir, exist_ok=True)

    full_output_folder, filename, counter, _subfolder, _filename_prefix = folder_paths.get_save_image_path(
        filename_prefix, latents_dir
    )

    extra_meta = sidecar_extra if isinstance(sidecar_extra, dict) else None
    meta = _build_latent_metadata(
        prompt=prompt,
        extra_pnginfo=extra_pnginfo,
        unique_id=unique_id,
        extra_meta=extra_meta,
    )

    latent_path = os.path.join(full_output_folder, f"{filename}_{counter:05}_.latent")
    payload = {
        "latent_tensor": samples["samples"].contiguous(),
        "latent_format_version_0": torch.tensor([]),
    }
    comfy.utils.save_torch_file(payload, latent_path, metadata=meta)

    sidecar = {"positive": positive, "negative": negative}
    if isinstance(sidecar_extra, dict):
        sidecar.update(sidecar_extra)
    torch.save(sidecar, latent_path.replace(".latent", ".cond.pt"))
    return latent_path


def _load_i2v_conditioning_sidecar(latent_path):
    cond_path = latent_path.replace(".latent", ".cond.pt")
    if not os.path.exists(cond_path):
        return [], [], {}

    try:
        data = torch.load(cond_path, map_location="cpu")
    except Exception:
        return [], [], {}

    if not isinstance(data, dict):
        return [], [], {}

    return data.get("positive", []), data.get("negative", []), data


def _coerce_trim_latent(value, default=0):
    try:
        if isinstance(value, str):
            parsed = _safe_json_loads(value)
            value = parsed if parsed is not None else value
        return int(value)
    except Exception:
        return int(default)


def _extract_prompt_text_from_ksampler(graph: Dict[str, Any], ks_node: Dict[str, Any]) -> Tuple[str, str]:
    pos = ""
    neg = ""

    inputs = ks_node.get("inputs", {}) if isinstance(ks_node, dict) else {}
    if not isinstance(inputs, dict):
        return pos, neg

    def _text_from_clip(link_value: Any) -> str:
        node_id = _linked_node_id(link_value)
        if node_id is None:
            return ""
        node = _get_graph_node(graph, node_id) or {}
        if node.get("class_type") == "CLIPTextEncode":
            return str(node.get("inputs", {}).get("text", "")).strip()
        return ""

    pos = _text_from_clip(inputs.get("positive"))
    neg = _text_from_clip(inputs.get("negative"))
    return pos, neg


def _extract_params_from_prompt_json(
    prompt_json: Dict[str, Any],
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, int, float, str, str, int]:
    """
    Returns: (positive, negative, steps, cfg, sampler_name, scheduler, end_at_step)
    parsed from the saved Comfy prompt graph with deterministic KSampler selection.
    """
    pos = ""
    neg = ""
    steps = 20
    cfg = 8.0
    sampler_name = ""
    scheduler = ""
    end_at_step = 0

    graph = _normalize_prompt_graph(prompt_json)
    if not isinstance(graph, dict):
        return pos, neg, steps, cfg, sampler_name, scheduler, end_at_step

    ks_node = None
    extracted_params: Dict[str, Any] = {}

    # 1) Source KSampler id saved directly in latent metadata.
    if isinstance(meta, dict):
        raw_ks = meta.get("mxd_source_ksampler_node_id")
        if raw_ks is not None:
            candidate = _get_graph_node(graph, str(raw_ks))
            if candidate and _is_ksampler_node(candidate):
                ks_node = candidate

    # 2) Source save node id -> trace upstream to nearest KSampler.
    if ks_node is None and isinstance(meta, dict):
        raw_save = meta.get("mxd_source_save_node_id")
        if raw_save is not None:
            save_node = _get_graph_node(graph, str(raw_save))
            if save_node:
                for start_id in _collect_upstream_linked_node_ids(save_node):
                    trace_id = _find_upstream_ksampler_node_id(graph, start_id)
                    if trace_id:
                        candidate = _get_graph_node(graph, trace_id)
                        if candidate and _is_ksampler_node(candidate):
                            ks_node = candidate
                            break

    # 3) Legacy fallback: last KSampler node in graph.
    if ks_node is None:
        for _, node in _iter_graph_nodes_sorted(graph):
            if _is_ksampler_node(node):
                ks_node = node

    if not ks_node:
        return pos, neg, steps, cfg, sampler_name, scheduler, end_at_step

    pos, neg = _extract_prompt_text_from_ksampler(graph, ks_node)
    extracted_params = _extract_ksampler_params(ks_node)

    if "steps" in extracted_params:
        steps = int(extracted_params["steps"])
    if "cfg" in extracted_params:
        cfg = float(extracted_params["cfg"])
    if "end_at_step" in extracted_params:
        end_at_step = int(extracted_params["end_at_step"])
    if "sampler_name" in extracted_params:
        sampler_name = str(extracted_params["sampler_name"]).strip()
    if "scheduler" in extracted_params:
        scheduler = str(extracted_params["scheduler"]).strip()

    # Fallback to saved parameter snapshot if graph parse is incomplete.
    if isinstance(meta, dict):
        saved_params = _safe_json_loads(meta.get("mxd_source_ksampler_params"))
        if isinstance(saved_params, dict):
            if "steps" in saved_params and "steps" not in extracted_params:
                try:
                    steps = int(saved_params["steps"])
                except Exception:
                    pass
            if "cfg" in saved_params and "cfg" not in extracted_params:
                try:
                    cfg = float(saved_params["cfg"])
                except Exception:
                    pass
            if "end_at_step" in saved_params and "end_at_step" not in extracted_params:
                try:
                    end_at_step = int(saved_params["end_at_step"])
                except Exception:
                    pass
            if "sampler_name" in saved_params and "sampler_name" not in extracted_params:
                try:
                    sampler_name = str(saved_params["sampler_name"]).strip()
                except Exception:
                    pass
            if "scheduler" in saved_params and "scheduler" not in extracted_params:
                try:
                    scheduler = str(saved_params["scheduler"]).strip()
                except Exception:
                    pass

    return pos, neg, steps, cfg, sampler_name, scheduler, end_at_step

def _coerce_enum(value, enum_values):
    try:
        return value if (enum_values and value in enum_values) else (enum_values[0] if enum_values else value)
    except Exception:
        return value


def _extract_sd3_shift(meta: dict, prompt_json: dict | None) -> float:
    """
    Find SD3 'shift' in several places:
    1) flat meta["shift"]
    2) nested in prompt/workflow JSON:
    - nodes[].{type|class_type} == "ModelSamplingSD3" -> inputs.shift or widgets_values[0]
    - runtime-style prompt dict mapping IDs -> {..., class_type: "ModelSamplingSD3"}
    Falls back to 5.0 if not found.
    """
    def try_float(x):
        try:
            return float(x)
        except Exception:
            return None

    # 1) flat meta
    if isinstance(meta, dict):
        v = try_float(meta.get("shift"))
        if v is not None:
            return v

    # parse any JSON-like strings present in meta
    def safe_load(x):
        try:
            return _safe_json_loads(x) if isinstance(x, str) else x
        except Exception:
            return None

    # Search helper over various JSON shapes
    def search_container(obj):
        # Direct dict containing shift
        if isinstance(obj, dict):
            if "shift" in obj:
                v = try_float(obj.get("shift"))
                if v is not None:
                    return v

            # Comfy "nodes": [ {...}, ... ]
            nodes = obj.get("nodes")
            if isinstance(nodes, list):
                # take the last SD3 node (most recent in graph)
                ms_nodes = [n for n in nodes if isinstance(n, dict) and (
                    n.get("type") == "ModelSamplingSD3" or
                    n.get("class_type") == "ModelSamplingSD3" or
                    (isinstance(n.get("properties"), dict) and n["properties"].get("Node name for S&R") == "ModelSamplingSD3")
                )]
                if ms_nodes:
                    nd = ms_nodes[-1]
                    # Prefer explicit inputs.shift if present and literal
                    inp = nd.get("inputs")
                    if isinstance(inp, dict) and "shift" in inp:
                        vv = inp["shift"]
                        # ignore connection like [node_id, idx]
                        if not isinstance(vv, (list, tuple)):
                            v2 = try_float(vv)
                            if v2 is not None:
                                return v2
                    # Fallback: first widget is shift for SD3 (as seen in your JSON)
                    w = nd.get("widgets_values")
                    if isinstance(w, list) and len(w) >= 1:
                        v2 = try_float(w[0])
                        if v2 is not None:
                            return v2

            # Runtime prompt map: {"42": {"class_type":"ModelSamplingSD3", "inputs":{...}, "widgets_values":[...]}, ...}
            # Heuristic: values that are dicts with class_type keys
            has_ct = [v for v in obj.values() if isinstance(v, dict) and "class_type" in v]
            if has_ct:
                for nd in has_ct:
                    if nd.get("class_type") == "ModelSamplingSD3":
                        inp = nd.get("inputs", {})
                        if isinstance(inp, dict) and "shift" in inp:
                            vv = inp["shift"]
                            if not isinstance(vv, (list, tuple)):
                                v2 = try_float(vv)
                                if v2 is not None:
                                    return v2
                        w = nd.get("widgets_values")
                        if isinstance(w, list) and len(w) >= 1:
                            v2 = try_float(w[0])
                            if v2 is not None:
                                return v2

        # Lists / nested
        if isinstance(obj, list):
            for it in obj:
                v = search_container(it)
                if v is not None:
                    return v
        return None

    # 2) Look in provided prompt_json
    v = search_container(prompt_json)
    if v is not None:
        return v

    # Also look in common meta fields that can hold the full workflow/prompt
    for key in ("workflow", "prompt", "extra_pnginfo"):
        candidate = meta.get(key)
        cand_obj = safe_load(candidate)
        if isinstance(cand_obj, dict) or isinstance(cand_obj, list):
            v = search_container(cand_obj)
            if v is not None:
                return v
        # extra_pnginfo can nest "workflow"/"prompt" again
        if isinstance(cand_obj, dict):
            for subkey in ("workflow", "prompt"):
                sub = safe_load(cand_obj.get(subkey))
                if isinstance(sub, dict) or isinstance(sub, list):
                    v = search_container(sub)
                    if v is not None:
                        return v

    # default
    return 5.0


# ---------- Load one latent (conditioning + sampler params + optional trim_latent) ----------
class LoadLatent_I2V_MXD:
    """
    Default single-latent loader: sampler settings, CONDITIONING (positive/negative), and
    an optional trim_latent value, all read from the .latent file and its .cond.pt sidecar.
    """
    DESCRIPTION = """Load one latent and return conditioning and sampler settings."""
    TITLE = "Load Latent MXD"
    CATEGORY = "MXD/Latents"
    FUNCTION = "load"

    RETURN_TYPES = (
        "FLOAT",         # shift
        "CONDITIONING",  # positive conditioning
        "CONDITIONING",  # negative conditioning
        "LATENT",
        "INT",
        "FLOAT",
        "STRING",
        "STRING",
        "INT",
        "STRING",
        "INT",           # trim_latent
        "STRING",        # high_workflow
    )
    RETURN_NAMES = (
        "shift",
        "positive",
        "negative",
        "samples",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "end_at_step",
        "filename_prefix",
        "trim_latent",
        "high_workflow",
    )

    @classmethod
    def INPUT_TYPES(s):
        latents_root = os.path.join(folder_paths.get_input_directory(), "latents")
        os.makedirs(latents_root, exist_ok=True)
        files = glob.glob(os.path.join(latents_root, "**", "*.latent"), recursive=True)
        # Clean dropdown display (no "latents/" prefix)
        options = [os.path.relpath(f, latents_root).replace(os.sep, "/") for f in files]
        # Group by folder so the combo's prev/next arrows walk one folder at a time.
        options = _sort_latent_options_by_folder(options, latents_root)

        ks_inputs = KSamplerAdvanced.INPUT_TYPES().get("required", {})
        samplers_enum   = ks_inputs.get("sampler_name", ("STRING",))[0]
        schedulers_enum = ks_inputs.get("scheduler",    ("STRING",))[0]

        s.RETURN_TYPES = (
            "FLOAT", "CONDITIONING", "CONDITIONING", "LATENT",
            "INT", "FLOAT", samplers_enum, schedulers_enum,
            "INT", "STRING", "INT", "STRING",
        )
        s._SAMPLERS_ENUM   = samplers_enum
        s._SCHEDULERS_ENUM = schedulers_enum

        return {
            "required": {
                "latent": (options, ),
                "run_folder": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When enabled, hitting Queue Prompt auto-queues every latent in this file's folder, one after another, instead of just the selected file.",
                }),
                "refresh_before_run": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When run_folder is on, re-scan the latents folder for new files right before the queuing loop starts, instead of using the dropdown list as of whenever it was last populated. Use this when another workflow is still writing latents into this folder as you queue this one.",
                }),
            }
        }

    @classmethod
    def IS_CHANGED(s, latent):
        # Fix path lookup (add "latents/" prefix back)
        p = folder_paths.get_annotated_filepath(f"latents/{latent}")
        m = hashlib.sha256()
        with open(p, "rb") as f:
            m.update(f.read())
        side = p.replace(".latent", ".cond.pt")
        if os.path.exists(side):
            with open(side, "rb") as f:
                m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, latent):
        check_path = latent if latent.startswith("latents/") else f"latents/{latent}"
        try:
            folder_paths.get_annotated_filepath(check_path)
        except Exception:
            return f"Invalid latent file: {latent}"
        return True

    def load(self, latent, run_folder=False, refresh_before_run=False):
        # Ensure we prepend "latents/" if missing, but don't duplicate it
        if not latent.startswith("latents/"):
            latent_path = folder_paths.get_annotated_filepath(f"latents/{latent}")
        else:
            latent_path = folder_paths.get_annotated_filepath(latent)

        sample_dict, meta, _ = _load_latent_file(latent_path)
        t = sample_dict["samples"]

        if isinstance(t, torch.Tensor) and t.dim() >= 4 and t.size(0) > 1:
            samples = {"samples": t[0:1].contiguous()}
        elif isinstance(t, torch.Tensor) and t.dim() >= 4 and t.size(0) == 1:
            samples = {"samples": t}
        else:
            samples = {"samples": t.unsqueeze(0)}

        prompt_json = _safe_json_loads(meta.get("prompt"))
        _pos_text, _neg_text, steps, cfg, sampler_name, scheduler, end_at_step = \
            _extract_params_from_prompt_json(prompt_json or {}, meta)

        # SD3 shift (not in KSamplerAdvanced, but we want it)
        shift = _extract_sd3_shift(meta, prompt_json)

        sampler_name = _coerce_enum(sampler_name, getattr(self.__class__, "_SAMPLERS_ENUM", ()))
        scheduler    = _coerce_enum(scheduler,    getattr(self.__class__, "_SCHEDULERS_ENUM", ()))

        def normalize_folder(part: str) -> str:
            part = part.replace("\\", "/").strip("/")
            if not part:
                return ""
            segments = [seg for seg in part.split("/") if seg]
            if segments and segments[0].lower() == "latents":
                segments = segments[1:]
            return "/".join(segments)

        folder_part = normalize_folder(os.path.dirname(latent))
        base_name   = os.path.basename(latent_path)
        clean_stem  = _strip_counter(base_name)
        prefix      = f"{folder_part}/{clean_stem}" if folder_part else clean_stem

        # Raw workflow JSON embedded when this latent was saved (empty string if none).
        source_workflow = meta.get("workflow") or ""

        positive_conditioning, negative_conditioning, sidecar = _load_i2v_conditioning_sidecar(latent_path)
        trim_latent = _coerce_trim_latent(sidecar.get("trim_latent", meta.get("trim_latent", 0)))

        return (
            float(shift),
            positive_conditioning,
            negative_conditioning,
            samples,
            int(steps),
            float(cfg),
            sampler_name,
            scheduler,
            int(end_at_step),
            prefix,
            trim_latent,
            source_workflow,
        )

# ---------- Load multiple latents from a folder (conditioning + sampler params + optional trim_latent) ----------
class LoadLatents_FromFolder_I2V_MXD:
    """
    Default folder/batch loader: same outputs as LoadLatent_I2V_MXD, one set per latent
    found in the folder.
    """
    DESCRIPTION = """Load all latents in a folder with conditioning and sampler settings."""
    TITLE = "Load Latent Batch MXD"
    CATEGORY = "MXD/Latents"
    FUNCTION = "load_batch_i2v"

    RETURN_TYPES = (
        "FLOAT",         # shift
        "CONDITIONING",  # positive conditioning
        "CONDITIONING",  # negative conditioning
        "LATENT",
        "INT",
        "FLOAT",
        "STRING",        # will be replaced with sampler enum in INPUT_TYPES
        "STRING",        # will be replaced with scheduler enum in INPUT_TYPES
        "INT",
        "STRING",
        "INT",           # trim_latent
    )
    RETURN_NAMES = (
        "shift",
        "positive",
        "negative",
        "samples",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "end_at_step",
        "filename_prefix",
        "trim_latent",
    )

    OUTPUT_IS_LIST = (True,) * 11

    @classmethod
    def INPUT_TYPES(s):
        # Same folder logic as the single loader
        latents_root = os.path.join(folder_paths.get_input_directory(), "latents")
        os.makedirs(latents_root, exist_ok=True)
        subs = [""] + _list_latent_subfolders(latents_root)

        # Pull live enums from KSamplerAdvanced so sampler/scheduler wire cleanly
        from nodes import KSamplerAdvanced
        ks_inputs = KSamplerAdvanced.INPUT_TYPES().get("required", {})
        samplers_enum   = ks_inputs.get("sampler_name", ("STRING",))[0]
        schedulers_enum = ks_inputs.get("scheduler",    ("STRING",))[0]

        s.RETURN_TYPES = (
            "FLOAT",         # shift
            "CONDITIONING",  # positive conditioning
            "CONDITIONING",  # negative conditioning
            "LATENT",
            "INT",
            "FLOAT",
            samplers_enum,   # enum type for sampler_name
            schedulers_enum, # enum type for scheduler
            "INT",
            "STRING",
            "INT",
        )
        s._SAMPLERS_ENUM   = samplers_enum
        s._SCHEDULERS_ENUM = schedulers_enum

        return {"required": {"subfolder": (subs, )}}

    def load_batch_i2v(self, subfolder):
        latents_root = os.path.join(folder_paths.get_input_directory(), "latents")
        base = os.path.join(latents_root, subfolder) if subfolder else latents_root
        files = glob.glob(os.path.join(base, "**", "*.latent"), recursive=True)
        files = _sort_paths_newest_first(files)
        if not files:
            raise RuntimeError(f"[LoadLatents_FromFolder_I2V_MXD] No .latent files found in '{base}'.")

        shifts, samples_list = [], []
        positives, negatives = [], []
        steps_list, cfgs, samplers, schedulers, end_steps = [], [], [], [], []
        filename_prefixes, trims = [], []

        for path in files:
            sample_dict, meta, _ = _load_latent_file(path)
            t = sample_dict["samples"]

            if isinstance(t, torch.Tensor) and t.dim() >= 4 and t.size(0) > 1:
                slices = [t[i:i+1].contiguous() for i in range(t.size(0))]
            else:
                slices = [t if (isinstance(t, torch.Tensor) and t.dim() >= 4 and t.size(0) == 1)
                          else t.unsqueeze(0)]

            prompt_json = _safe_json_loads(meta.get("prompt"))
            _pos_text, _neg_text, n_steps, cfg, sampler_name, scheduler, end_at_step = \
                _extract_params_from_prompt_json(prompt_json or {}, meta)

            sampler_name = _coerce_enum(sampler_name, getattr(self.__class__, "_SAMPLERS_ENUM", ()))
            scheduler    = _coerce_enum(scheduler,    getattr(self.__class__, "_SCHEDULERS_ENUM", ()))
            shift_val    = _extract_sd3_shift(meta, prompt_json)

            positive_conditioning, negative_conditioning, sidecar = _load_i2v_conditioning_sidecar(path)
            trim_latent = _coerce_trim_latent(sidecar.get("trim_latent", meta.get("trim_latent", 0)))

            folder_part = subfolder if subfolder else ""
            clean_stem  = _strip_counter(os.path.basename(path))
            prefix      = os.path.join(folder_part, clean_stem) if folder_part else clean_stem

            for sl in slices:
                shifts.append(float(shift_val))
                positives.append(positive_conditioning)
                negatives.append(negative_conditioning)
                samples_list.append({"samples": sl})
                steps_list.append(int(n_steps))
                cfgs.append(float(cfg))
                samplers.append(sampler_name)
                schedulers.append(scheduler)
                end_steps.append(int(end_at_step))
                filename_prefixes.append(prefix)
                trims.append(trim_latent)

        return (
            shifts,
            positives,
            negatives,
            samples_list,
            steps_list,
            cfgs,
            samplers,
            schedulers,
            end_steps,
            filename_prefixes,
            trims,
        )

# ---------- Pipe variants: bundle all the loader outputs into one wire ----------
class LoadLatent_I2V_Pipe_MXD(LoadLatent_I2V_MXD):
    """
    Same loading logic as LoadLatent_I2V_MXD, but bundles every value into a single
    MXD_LATENT_PIPE output so switching between the single/batch loaders is a one-wire swap.
    Unpack with LatentPipeUnpack_MXD.
    """
    TITLE = "Load Latent Pipe MXD"
    CATEGORY = "MXD/Latents"
    FUNCTION = "load_pipe"

    RETURN_TYPES = ("MXD_LATENT_PIPE",)
    RETURN_NAMES = ("latent_pipe",)

    @classmethod
    def INPUT_TYPES(s):
        inputs = LoadLatent_I2V_MXD.INPUT_TYPES.__func__(s)
        s.RETURN_TYPES = ("MXD_LATENT_PIPE",)
        return inputs

    def load_pipe(self, latent, run_folder=False, refresh_before_run=False):
        (
            shift, positive, negative, samples,
            steps, cfg, sampler_name, scheduler,
            end_at_step, prefix, trim_latent, source_workflow,
        ) = self.load(latent, run_folder, refresh_before_run)

        pipe = {
            "shift": shift,
            "positive": positive,
            "negative": negative,
            "samples": samples,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "end_at_step": end_at_step,
            "filename_prefix": prefix,
            "trim_latent": trim_latent,
            "high_workflow": source_workflow,
        }
        return (pipe,)


class LoadLatents_FromFolder_I2V_Pipe_MXD(LoadLatents_FromFolder_I2V_MXD):
    """
    Same loading logic as LoadLatents_FromFolder_I2V_MXD, but bundles every value into a
    single MXD_LATENT_PIPE output per item. Unpack with LatentPipeUnpack_MXD.
    """
    TITLE = "Load Latent Batch Pipe MXD"
    CATEGORY = "MXD/Latents"
    FUNCTION = "load_batch_pipe"

    RETURN_TYPES = ("MXD_LATENT_PIPE",)
    RETURN_NAMES = ("latent_pipe",)
    OUTPUT_IS_LIST = (True,)

    @classmethod
    def INPUT_TYPES(s):
        inputs = LoadLatents_FromFolder_I2V_MXD.INPUT_TYPES.__func__(s)
        s.RETURN_TYPES = ("MXD_LATENT_PIPE",)
        return inputs

    def load_batch_pipe(self, subfolder):
        (
            shifts, positives, negatives, samples_list,
            steps_list, cfgs, samplers, schedulers,
            end_steps, filename_prefixes, trims,
        ) = self.load_batch_i2v(subfolder)

        pipes = []
        for i in range(len(samples_list)):
            pipes.append({
                "shift": shifts[i],
                "positive": positives[i],
                "negative": negatives[i],
                "samples": samples_list[i],
                "steps": steps_list[i],
                "cfg": cfgs[i],
                "sampler_name": samplers[i],
                "scheduler": schedulers[i],
                "end_at_step": end_steps[i],
                "filename_prefix": filename_prefixes[i],
                "trim_latent": trims[i],
            })
        return (pipes,)


class LatentPipeUnpack_MXD:
    """
    Splits an MXD_LATENT_PIPE back into shift, conditioning, samples, and sampler settings.
    Works with any MXD latent pipe loader (single or batch, I2V or VACE 2.2) - missing
    fields like trim_latent just fall back to a safe default.
    """
    DESCRIPTION = """Split a latent pipe back into shift, positive, negative, samples, and sampler settings."""
    TITLE = "Unpack Latent Pipe MXD"
    CATEGORY = "MXD/Latents"
    FUNCTION = "unpack"

    RETURN_TYPES = (
        "FLOAT", "CONDITIONING", "CONDITIONING", "LATENT",
        "INT", "FLOAT", "STRING", "STRING", "INT", "STRING", "INT", "STRING",
    )
    RETURN_NAMES = (
        "shift", "positive", "negative", "samples",
        "steps", "cfg", "sampler_name", "scheduler",
        "end_at_step", "filename_prefix", "trim_latent", "high_workflow",
    )

    @classmethod
    def INPUT_TYPES(s):
        from nodes import KSamplerAdvanced
        ks_inputs = KSamplerAdvanced.INPUT_TYPES().get("required", {})
        samplers_enum   = ks_inputs.get("sampler_name", ("STRING",))[0]
        schedulers_enum = ks_inputs.get("scheduler",    ("STRING",))[0]

        s.RETURN_TYPES = (
            "FLOAT", "CONDITIONING", "CONDITIONING", "LATENT",
            "INT", "FLOAT", samplers_enum, schedulers_enum,
            "INT", "STRING", "INT", "STRING",
        )
        s._SAMPLERS_ENUM = samplers_enum
        s._SCHEDULERS_ENUM = schedulers_enum

        return {"required": {"latent_pipe": ("MXD_LATENT_PIPE",)}}

    def unpack(self, latent_pipe):
        sampler_name = _coerce_enum(latent_pipe.get("sampler_name"), getattr(self.__class__, "_SAMPLERS_ENUM", ()))
        scheduler    = _coerce_enum(latent_pipe.get("scheduler"),    getattr(self.__class__, "_SCHEDULERS_ENUM", ()))

        return (
            latent_pipe.get("shift", 0.0),
            latent_pipe.get("positive", []),
            latent_pipe.get("negative", []),
            latent_pipe.get("samples"),
            latent_pipe.get("steps", 0),
            latent_pipe.get("cfg", 0.0),
            sampler_name,
            scheduler,
            latent_pipe.get("end_at_step", 0),
            latent_pipe.get("filename_prefix", ""),
            latent_pipe.get("trim_latent", 0),
            latent_pipe.get("high_workflow", ""),
        )


NODE_CLASS_MAPPINGS = {
    "SaveLatent_I2V_MXD": SaveLatent_I2V_MXD,
    "LoadLatent_I2V_MXD": LoadLatent_I2V_MXD,
    "LoadLatents_FromFolder_I2V_MXD": LoadLatents_FromFolder_I2V_MXD,
    "LoadLatent_I2V_Pipe_MXD": LoadLatent_I2V_Pipe_MXD,
    "LoadLatents_FromFolder_I2V_Pipe_MXD": LoadLatents_FromFolder_I2V_Pipe_MXD,
    "LatentPipeUnpack_MXD": LatentPipeUnpack_MXD,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveLatent_I2V_MXD": "Save Latent MXD",
    "LoadLatent_I2V_MXD": "Load Latent MXD",
    "LoadLatents_FromFolder_I2V_MXD": "Load Latent Batch MXD",
    "LoadLatent_I2V_Pipe_MXD": "Load Latent Pipe MXD",
    "LoadLatents_FromFolder_I2V_Pipe_MXD": "Load Latent Batch Pipe MXD",
    "LatentPipeUnpack_MXD": "Unpack Latent Pipe MXD",
}

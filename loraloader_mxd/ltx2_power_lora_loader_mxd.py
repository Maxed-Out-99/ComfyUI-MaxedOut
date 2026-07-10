import folder_paths
import comfy.utils
import comfy.lora

from .constants import get_category
from .power_prompt_utils import get_lora_by_filename
from .utils import FlexibleOptionalInputType, any_type
from .log import log_node_warn

NODE_NAME = "LTX2 Lora Loader MXD"

# Layer-name substrings, checked most-specific first, mirroring KJNodes' LTX2 LoRA Loader Advanced.
LAYER_STRENGTH_KEY_ORDER = (
  ("video_to_audio", ("video_to_audio_attn",)),
  ("audio_to_video", ("audio_to_video_attn",)),
  ("audio", ("audio_attn", "audio_ff.net")),
  ("video", ("attn", "ff.net")),
)
DEFAULT_LAYER_STRENGTHS = {
  "video": 1.0,
  "video_to_audio": 1.0,
  "audio": 1.0,
  "audio_to_video": 1.0,
  "other": 1.0,
}


class MxdLtx2PowerLoraLoader:
  """Stacked LoRA loader for LTX2 models.

  Combines the multi-LoRA stack UI of Lora Loader MXD with the per-layer-type strength
  shaping (video / audio / cross-attention / other) and optional per-DiT-block ratios from
  KJNodes' LTX2 LoRA Loader Advanced. Every enabled LoRA in the stack is shaped by the same
  layer/block controls before being patched onto the model.
  """

  NAME = NODE_NAME
  CATEGORY = get_category()

  @classmethod
  def INPUT_TYPES(cls):  # pylint: disable=invalid-name,missing-function-docstring
    return {
      "required": {
        "model": ("MODEL",),
      },
      "optional": FlexibleOptionalInputType(type=any_type, data={
        "blocks": ("SELECTEDDITBLOCKS",),
      }),
      "hidden": {},
    }

  RETURN_TYPES = ("MODEL", "STRING", "STRING")
  RETURN_NAMES = ("MODEL", "rank", "loaded_keys_info")
  FUNCTION = "load_loras"

  @staticmethod
  def _coerce_bool(value, default=False) -> bool:
    if isinstance(value, bool):
      return value
    if isinstance(value, str):
      lowered = value.strip().lower()
      if lowered in {"true", "1", "yes", "on"}:
        return True
      if lowered in {"false", "0", "no", "off"}:
        return False
    if value is None:
      return default
    return bool(value)

  @staticmethod
  def _coerce_float(value, default=0.0) -> float:
    if isinstance(value, bool):
      return float(value)
    try:
      if value is None:
        return float(default)
      return float(value)
    except (TypeError, ValueError):
      return float(default)

  @classmethod
  def _layer_strength_multiplier(cls, key_str, layer_strengths):
    for name, needles in LAYER_STRENGTH_KEY_ORDER:
      if any(needle in key_str for needle in needles):
        return layer_strengths.get(name, 1.0)
    return layer_strengths.get("other", 1.0)

  @classmethod
  def _apply_shaping(cls, loaded, blocks, layer_strengths):
    """Applies block-ratio and layer-type strength shaping to a loaded LoRA's patch dict in place."""
    keys_to_delete = []

    if blocks:
      for block, ratio in blocks.items():
        for key in list(loaded.keys()):
          key_str = key if isinstance(key, str) else " ".join(k for k in key if isinstance(k, str))
          if block not in key_str:
            continue
          if ratio == 0:
            keys_to_delete.append(key)
          else:
            value = loaded[key]
            if hasattr(value, "weights"):
              weights_list = list(value.weights)
              weights_list[2] = ratio
              value.weights = tuple(weights_list)

    for key in list(loaded.keys()):
      if key in keys_to_delete:
        continue
      key_str = key if isinstance(key, str) else (key[0] if isinstance(key, tuple) else str(key))
      multiplier = cls._layer_strength_multiplier(key_str, layer_strengths)

      if multiplier == 0:
        keys_to_delete.append(key)
      elif multiplier != 1.0:
        value = loaded[key]
        if hasattr(value, "weights"):
          weights_list = list(value.weights)
          current_alpha = weights_list[2] if weights_list[2] is not None else 1.0
          weights_list[2] = current_alpha * multiplier
          value.weights = tuple(weights_list)

    for key in keys_to_delete:
      loaded.pop(key, None)

    return loaded

  def load_loras(self, model, blocks=None, **kwargs):
    layer_strengths = dict(DEFAULT_LAYER_STRENGTHS)
    lora_entries = []

    for key, value in kwargs.items():
      if not isinstance(value, dict):
        continue
      if value.get("type") == "Ltx2StrengthWidget":
        strength_key = value.get("key")
        if strength_key in layer_strengths:
          layer_strengths[strength_key] = self._coerce_float(value.get("value"), default=1.0)
        continue
      key_upper = key.upper()
      if not key_upper.startswith("LORA_"):
        continue
      if not all(k in value for k in ("on", "lora", "strength")):
        log_node_warn(NODE_NAME, f'Skipping malformed LoRA input "{key}" (missing fields).')
        continue
      lora_entries.append(value)

    key_map = {}
    if model is not None:
      key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)

    new_modelpatcher = model.clone() if model is not None else None
    rank_lines = []
    loaded_keys_lines = []

    for entry in lora_entries:
      if not self._coerce_bool(entry.get("on"), default=False):
        continue
      strength_model = self._coerce_float(entry.get("strength"), default=0.0)
      if strength_model == 0.0:
        continue

      lora_name = get_lora_by_filename(entry["lora"], log_node=NODE_NAME)
      if lora_name is None or new_modelpatcher is None:
        continue

      lora_path = folder_paths.get_full_path("loras", lora_name)
      if not lora_path:
        continue

      try:
        lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
      except Exception as exc:
        log_node_warn(NODE_NAME, f'Failed to load LoRA "{lora_name}" ({exc}). Skipping.')
        continue

      weight_key = next((k for k in lora_sd.keys() if k.endswith("weight")), None)
      rank = str(lora_sd[weight_key].shape[0]) if weight_key is not None else "unknown"
      rank_lines.append(f"{lora_name}: rank={rank}")

      loaded = comfy.lora.load_lora(lora_sd, key_map)
      loaded = self._apply_shaping(loaded, blocks, layer_strengths)

      if not loaded:
        loaded_keys_lines.append(f"{lora_name}: no matching keys after shaping.")
        continue

      applied = new_modelpatcher.add_patches(loaded, strength_model)
      applied = set(applied)
      for k in loaded:
        k_str = k if isinstance(k, str) else str(k)
        status = "loaded" if k in applied else "NOT LOADED"
        loaded_keys_lines.append(f"{lora_name} | {k_str}: {status}")

    result_model = new_modelpatcher if new_modelpatcher is not None else model
    rank_info = "\n".join(rank_lines) if rank_lines else "unknown"
    loaded_keys_info = "\n".join(loaded_keys_lines)

    return (result_model, rank_info, loaded_keys_info)

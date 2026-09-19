import folder_paths
import comfy.sd
import comfy.utils

from typing import Union

from nodes import LoraLoader
from .constants import get_category
from .power_prompt_utils import get_lora_by_filename
from .utils import FlexibleOptionalInputType, any_type
from .server.utils_info import get_model_info_file_data
from .log import log_node_warn

NODE_NAME = "Lora Loader MXD"


class MxdPowerLoraLoader:
  """Standalone Power LoRA Loader extracted from rgthree-comfy."""

  NAME = NODE_NAME
  CATEGORY = get_category()

  @classmethod
  def INPUT_TYPES(cls):  # pylint: disable=invalid-name,missing-function-docstring
    return {
      "required": {},
      "optional": FlexibleOptionalInputType(type=any_type, data={
        "model": ("MODEL",),
        "clip": ("CLIP",),
      }),
      "hidden": {},
    }

  RETURN_TYPES = ("MODEL", "CLIP")
  RETURN_NAMES = ("MODEL", "CLIP")
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
  def VALIDATE_INPUTS(cls, input_types, **kwargs):  # pylint: disable=invalid-name
    """Reject missing enabled LoRAs during prompt validation, before execution."""
    clip_connected = "clip" in input_types
    model_connected = "model" in input_types
    lora_paths = None

    for key, value in kwargs.items():
      if not key.upper().startswith("LORA_"):
        continue
      if not isinstance(value, dict):
        return f'{NODE_NAME}: malformed LoRA input "{key}" (expected object).'
      if not all(field in value for field in ("on", "lora", "strength")):
        if cls._coerce_bool(value.get("on"), default=False):
          return f'{NODE_NAME}: malformed LoRA input "{key}" (missing fields).'
        continue

      strength_model = cls._coerce_float(value.get("strength"), default=0.0)
      strength_clip = (
        cls._coerce_float(value.get("strengthTwo"), default=strength_model)
        if clip_connected
        else 0.0
      )
      if not cls._coerce_bool(value.get("on"), default=False):
        continue
      if strength_model == 0.0 and strength_clip == 0.0:
        continue

      lora_name = str(value.get("lora") or "").strip()
      if not lora_name:
        return f'{NODE_NAME}: enabled LoRA slot "{key}" has an empty filename.'
      if not model_connected:
        return f'{NODE_NAME}: LoRA "{lora_name}" is enabled but no MODEL is connected.'

      if lora_paths is None:
        lora_paths = folder_paths.get_filename_list("loras")
      if get_lora_by_filename(lora_name, lora_paths=lora_paths, log_node=None) is None:
        return (
          f'{NODE_NAME}: LoRA not found: "{lora_name}". '
          "Choose an installed LoRA or turn this row off."
        )

    return True

  def _apply_lora_without_clip(self, model, lora, strength_model, strength_clip):
    # Match stock ComfyUI: missing file must hard-fail, not silently no-op.
    get_path = getattr(folder_paths, "get_full_path_or_raise", None)
    if get_path is not None:
      lora_path = get_path("loras", lora)
    else:
      lora_path = folder_paths.get_full_path("loras", lora)
      if not lora_path:
        raise FileNotFoundError(f'LoRA not found: "{lora}"')
    loaded_lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
    model, _ = comfy.sd.load_lora_for_models(model, None, loaded_lora, strength_model, strength_clip)
    return model

  def load_loras(self, model=None, clip=None, **kwargs):
    for key, value in kwargs.items():
      key = key.upper()
      if not key.startswith("LORA_"):
        continue
      if not isinstance(value, dict):
        # Disabled/empty UI slots can arrive weirdly — only soft-skip junk that is off/empty.
        # Anything clearly toggled on must hard-fail like stock Loaders.
        raise ValueError(f'{NODE_NAME}: malformed LoRA input "{key}" (expected object).')
      if not all(k in value for k in ("on", "lora", "strength")):
        if self._coerce_bool(value.get("on"), default=False):
          raise ValueError(f'{NODE_NAME}: malformed LoRA input "{key}" (missing fields).')
        continue

      strength_model = self._coerce_float(value.get("strength"), default=0.0)
      strength_clip_raw = value.get("strengthTwo")

      if clip is None:
        if strength_clip_raw is not None and self._coerce_float(strength_clip_raw, 0.0) != 0.0:
          log_node_warn(NODE_NAME, "Received clip strength even though no clip supplied.")
        strength_clip = 0.0
      else:
        strength_clip = self._coerce_float(strength_clip_raw, default=strength_model)

      # Off / zero strength = intentionally unused slot (same as leaving a stock loader unused)
      if not self._coerce_bool(value.get("on"), default=False):
        continue
      if strength_model == 0.0 and strength_clip == 0.0:
        continue

      lora_name = value.get("lora") or ""
      if not str(lora_name).strip():
        raise FileNotFoundError(f'{NODE_NAME}: enabled LoRA slot has empty filename.')

      if model is None:
        raise RuntimeError(
          f'{NODE_NAME}: LoRA "{lora_name}" is enabled but no MODEL is connected.'
        )

      lora = get_lora_by_filename(lora_name, log_node=self.NAME)
      if lora is None:
        # Stock Load LoRA / Checkpoint behavior: missing file aborts the prompt.
        raise FileNotFoundError(
          f'{NODE_NAME}: LoRA not found: "{lora_name}". '
          f'Fix the slot or turn it off — refusing to continue silently.'
        )

      # Do not swallow apply errors — same as stock LoraLoader.
      if clip is None:
        model = self._apply_lora_without_clip(model, lora, strength_model, strength_clip)
      else:
        model, clip = LoraLoader().load_lora(model, clip, lora, strength_model, strength_clip)

    return (model, clip)

  @classmethod
  def get_enabled_loras_from_prompt_node(
    cls,
    prompt_node: dict,
  ) -> list[dict[str, Union[str, float]]]:
    result = []
    for name, lora in prompt_node["inputs"].items():
      if name.startswith("lora_") and lora["on"]:
        lora_file = get_lora_by_filename(lora["lora"], log_node=cls.NAME)
        if lora_file is not None:
          lora_dict = {
            "name": lora["lora"],
            "strength": lora["strength"],
            "path": folder_paths.get_full_path("loras", lora_file),
          }
          if "strengthTwo" in lora:
            lora_dict["strength_clip"] = lora["strengthTwo"]
          result.append(lora_dict)
    return result

  @classmethod
  def get_enabled_triggers_from_prompt_node(cls, prompt_node: dict, max_each: int = 1):
    loras = [l["name"] for l in cls.get_enabled_loras_from_prompt_node(prompt_node)]
    trained_words = []
    for lora in loras:
      info = get_model_info_file_data(lora, "loras", default={})
      if not info or not info.keys():
        log_node_warn(
          NODE_NAME,
          f"No info found for LoRA {lora} when grabbing triggers. Open the info dialog first.",
        )
        continue
      if "trainedWords" not in info or not info["trainedWords"]:
        log_node_warn(
          NODE_NAME,
          f"No trained words for LoRA {lora} when grabbing triggers.",
        )
        continue
      trained_words += [w for wi in info["trainedWords"][:max_each] if (wi and (w := wi["word"]))]
    return trained_words

from .power_lora_loader_mxd import MxdPowerLoraLoader
from .server import routes_model_info as _routes_model_info  # noqa: F401

NODE_CLASS_MAPPINGS = {
  MxdPowerLoraLoader.NAME: MxdPowerLoraLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
  MxdPowerLoraLoader.NAME: "Lora Loader MXD",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

from .power_lora_loader_mxd import MxdPowerLoraLoader
from .ltx2_power_lora_loader_mxd import MxdLtx2PowerLoraLoader
from .server import routes_model_info as _routes_model_info  # noqa: F401

NODE_CLASS_MAPPINGS = {
  MxdPowerLoraLoader.NAME: MxdPowerLoraLoader,
  MxdLtx2PowerLoraLoader.NAME: MxdLtx2PowerLoraLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
  MxdPowerLoraLoader.NAME: "Lora Loader MXD",
  MxdLtx2PowerLoraLoader.NAME: "LTX2 Lora Loader MXD",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

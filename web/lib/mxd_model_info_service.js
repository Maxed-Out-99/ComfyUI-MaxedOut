import { mxdApi } from "./mxd_api.js";
import { api } from "../../../scripts/api.js";

class BaseModelInfoService extends EventTarget {
  constructor(modelInfoType, apiRefreshEventString) {
    super();
    if (modelInfoType) this.modelInfoType = modelInfoType;
    if (apiRefreshEventString) this.apiRefreshEventString = apiRefreshEventString;
    this.fileToInfo = new Map();
    this.init();
  }

  init() {
    api.addEventListener(this.apiRefreshEventString, this.handleAsyncUpdate.bind(this));
  }

  async getInfo(file, refresh, light) {
    if (this.fileToInfo.has(file) && !refresh) {
      return this.fileToInfo.get(file);
    }
    return this.fetchInfo(file, refresh, light);
  }

  async refreshInfo(file) {
    return this.fetchInfo(file, true);
  }

  async clearFetchedInfo(file) {
    await mxdApi.clearModelsInfo({ type: this.modelInfoType, files: [file] });
    this.fileToInfo.delete(file);
    return null;
  }

  async savePartialInfo(file, data) {
    const info = await mxdApi.saveModelInfo(this.modelInfoType, file, data);
    this.fileToInfo.set(file, info);
    return info;
  }

  handleAsyncUpdate(event) {
    const info = event.detail?.data;
    if (info?.file) {
      this.fileToInfo.set(info.file, info);
    }
  }

  async fetchInfo(file, refresh = false, light = false) {
    let info = null;
    if (!refresh) {
      info = await mxdApi.getModelsInfo({ type: this.modelInfoType, files: [file], light });
    } else {
      info = await mxdApi.refreshModelsInfo({ type: this.modelInfoType, files: [file] });
    }
    info = info?.[0] ?? null;
    if (!light) {
      this.fileToInfo.set(file, info);
    }
    return info;
  }
}

class LoraInfoService extends BaseModelInfoService {
  apiRefreshEventString = "loraloader-mxd-refreshed-loras-info";
  modelInfoType = "loras";
}

class CheckpointInfoService extends BaseModelInfoService {
  apiRefreshEventString = "loraloader-mxd-refreshed-checkpoints-info";
  modelInfoType = "checkpoints";
}

/** Dispatches to a per-file-type `BaseModelInfoService`, for loader nodes (like the Smart
 * UNET/CLIP loaders) whose combo list mixes files that live under different folder_paths
 * keys (e.g. plain .safetensors vs .gguf) depending on the chosen file's extension. */
class DynamicModelInfoService {
  constructor(resolveType) {
    this.resolveType = resolveType;
    this.servicesByType = new Map();
  }

  _serviceFor(file) {
    const type = this.resolveType(file);
    if (!this.servicesByType.has(type)) {
      this.servicesByType.set(type, new BaseModelInfoService(type, `loraloader-mxd-refreshed-${type}-info`));
    }
    return this.servicesByType.get(type);
  }

  getInfo(file, refresh, light) {
    return this._serviceFor(file).getInfo(file, refresh, light);
  }

  refreshInfo(file) {
    return this._serviceFor(file).refreshInfo(file);
  }

  clearFetchedInfo(file) {
    return this._serviceFor(file).clearFetchedInfo(file);
  }

  savePartialInfo(file, data) {
    return this._serviceFor(file).savePartialInfo(file, data);
  }
}

export function resolveUnetModelType(file) {
  return String(file).toLowerCase().endsWith(".gguf") ? "unet_gguf" : "diffusion_models";
}

export function resolveClipModelType(file) {
  return String(file).toLowerCase().endsWith(".gguf") ? "clip_gguf" : "text_encoders";
}

export const LORA_INFO_SERVICE = new LoraInfoService();
export const CHECKPOINT_INFO_SERVICE = new CheckpointInfoService();
export const UNET_INFO_SERVICE = new DynamicModelInfoService(resolveUnetModelType);
export const CLIP_INFO_SERVICE = new DynamicModelInfoService(resolveClipModelType);

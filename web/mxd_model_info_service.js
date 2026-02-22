import { mxdApi } from "./mxd_api.js";
import { api } from "../../scripts/api.js";

class BaseModelInfoService extends EventTarget {
  constructor() {
    super();
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

export const LORA_INFO_SERVICE = new LoraInfoService();
export const CHECKPOINT_INFO_SERVICE = new CheckpointInfoService();

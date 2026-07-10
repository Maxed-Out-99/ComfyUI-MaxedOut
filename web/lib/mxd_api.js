class MxdApi {
  constructor(baseUrl) {
    this.getLorasPromise = null;
    this.filenamesPromises = {};
    this.setBaseUrl(baseUrl);
  }

  setBaseUrl(baseUrlArg) {
    let baseUrl = null;
    if (baseUrlArg) {
      baseUrl = baseUrlArg;
    } else if (window.location.pathname.includes("/loraloader-mxd/")) {
      const parts = window.location.pathname.split("/loraloader-mxd/")[1]?.split("/");
      if (parts && parts.length) {
        baseUrl = parts.map(() => "../").join("") + "loraloader-mxd/api";
      }
    }
    this.baseUrl = baseUrl || "./loraloader-mxd/api";
    const comfyBasePathname = location.pathname.includes("/loraloader-mxd/")
      ? location.pathname.split("loraloader-mxd/")[0]
      : location.pathname;
    this.comfyBaseUrl = comfyBasePathname.split("/").slice(0, -1).join("/");
  }

  apiURL(route) {
    return `${this.baseUrl}${route}`;
  }

  fetchApi(route, options) {
    return fetch(this.apiURL(route), options);
  }

  async fetchJson(route, options) {
    const r = await this.fetchApi(route, options);
    return await r.json();
  }

  async postJson(route, json) {
    const body = new FormData();
    body.append("json", JSON.stringify(json));
    return await this.fetchJson(route, { method: "POST", body });
  }

  getLoras(force = false) {
    if (!this.getLorasPromise || force) {
      this.getLorasPromise = this.fetchJson("/loras?format=details", { cache: "no-store" });
    }
    return this.getLorasPromise;
  }

  getFilenames(type, force = false) {
    if (!this.filenamesPromises[type] || force) {
      this.filenamesPromises[type] = this.fetchJson(`/${type}`, { cache: "no-store" });
    }
    return this.filenamesPromises[type];
  }

  async getUnetModelList(force = false) {
    const [standard, gguf] = await Promise.all([
      this.getFilenames("diffusion_models", force),
      this.getFilenames("unet_gguf", force),
    ]);
    return [...new Set([...standard, ...gguf])].sort();
  }

  async getClipModelList(force = false) {
    const [standard, gguf] = await Promise.all([
      this.getFilenames("text_encoders", force),
      this.getFilenames("clip_gguf", force),
    ]);
    return [...new Set([...standard, ...gguf])].sort();
  }

  getCheckpointList(force = false) {
    return this.getFilenames("checkpoints", force);
  }

  async fetchApiJsonOrNull(route, options) {
    const response = await this.fetchJson(route, options);
    if (response.status === 200 && response.data) {
      return response.data || null;
    }
    return null;
  }

  async getModelsInfo(options) {
    const params = new URLSearchParams();
    if (options.files?.length) {
      params.set("files", options.files.join(","));
    }
    if (options.light) {
      params.set("light", "1");
    }
    if (options.format) {
      params.set("format", options.format);
    }
    const path = `/${options.type}/info?` + params.toString();
    return (await this.fetchApiJsonOrNull(path)) || [];
  }

  async refreshModelsInfo(options) {
    const params = new URLSearchParams();
    if (options.files?.length) {
      params.set("files", options.files.join(","));
    }
    const path = `/${options.type}/info/refresh?` + params.toString();
    return await this.fetchApiJsonOrNull(path);
  }

  async clearModelsInfo(options) {
    const params = new URLSearchParams();
    if (options.files?.length) {
      params.set("files", options.files.join(","));
    }
    const path = `/${options.type}/info/clear?` + params.toString();
    await this.fetchApiJsonOrNull(path);
  }

  async saveModelInfo(type, file, data) {
    const body = new FormData();
    body.append("json", JSON.stringify(data));
    return await this.fetchApiJsonOrNull(`/${type}/info?file=${encodeURIComponent(file)}`, {
      cache: "no-store",
      method: "POST",
      body,
    });
  }

  fetchComfyApi(route, options) {
    const url = this.comfyBaseUrl + "/api" + route;
    const opts = options || {};
    opts.headers = opts.headers || {};
    opts.cache = opts.cache || "no-cache";
    return fetch(url, opts);
  }
}

export const mxdApi = new MxdApi();

import { MxdDialog } from "./mxd_dialog.js";
import {
  createElement as $el,
  empty,
  appendChildren,
  getClosestOrSelf,
  query,
  queryAll,
  setAttributes,
} from "./mxd_utils_dom.js";
import { logoCivitai, link, pencilColored, diskColored, dotdotdot } from "./mxd_svgs.js";
import { CHECKPOINT_INFO_SERVICE, LORA_INFO_SERVICE } from "./mxd_model_info_service.js";
import { mxdRuntime } from "./mxd_runtime.js";
import { MenuButton } from "./mxd_menu.js";
import { generateId, injectCss } from "./mxd_shared_utils.js";

const EXTENSION_BASE = new URL(".", import.meta.url).pathname.replace(/\/$/, "");
class MxdInfoDialog extends MxdDialog {
  constructor(file) {
    const dialogOptions = {
      class: "rgthree-info-dialog",
      title: `<h2>Loading...</h2>`,
      content: "<center>Loading..</center>",
      onBeforeClose: () => true,
    };
    super(dialogOptions);
    this.modifiedModelData = false;
    this.modelInfo = null;
    this.init(file);
  }

  async init(file) {
    await injectCss(`${EXTENSION_BASE}/mxd_dialog_base.css`);
    await injectCss(`${EXTENSION_BASE}/mxd_dialog_model_info.css`);
    this.modelInfo = await this.getModelInfo(file);
    this.setContent(this.getInfoContent());
    this.setTitle(this.modelInfo?.name || this.modelInfo?.file || "Unknown");
    this.attachEvents();
  }

  getCloseEventDetail() {
    return { detail: { dirty: this.modifiedModelData } };
  }

  attachEvents() {
    this.contentElement.addEventListener("click", async (e) => {
      const target = getClosestOrSelf(e.target, "[data-action]");
      const action = target?.getAttribute("data-action");
      if (!target || !action) return;
      await this.handleEventAction(action, target, e);
    });
  }

  async handleEventAction(action, target, e) {
    const info = this.modelInfo;
    if (!info?.file) return;

    if (action === "fetch-civitai") {
      this.modelInfo = await this.refreshModelInfo(info.file);
      this.setContent(this.getInfoContent());
      this.setTitle(this.modelInfo?.name || this.modelInfo?.file || "Unknown");
    } else if (action === "copy-trained-words") {
      const selected = queryAll(".-rgthree-is-selected", target.closest("tr"));
      const text = selected.map((el) => el.getAttribute("data-word")).join(", ");
      await navigator.clipboard.writeText(text);
      mxdRuntime.showMessage({
        id: "copy-trained-words-" + generateId(4),
        type: "success",
        message: `Copied ${selected.length} key word${selected.length === 1 ? "" : "s"}.`,
        timeout: 3000,
      });
    } else if (action === "toggle-trained-word") {
      target?.classList.toggle("-rgthree-is-selected");
      const tr = target.closest("tr");
      if (tr) {
        const span = query("td:first-child > *", tr);
        let small = query("small", span);
        if (!small) {
          small = $el("small", { parent: span });
        }
        const num = queryAll(".-rgthree-is-selected", tr).length;
        small.innerHTML = num ? `${num} selected | <span role="button" data-action="copy-trained-words">Copy</span>` : "";
      }
    } else if (action === "edit-row") {
      const tr = target.closest("tr");
      const td = query("td:nth-child(2)", tr);
      const input = td.querySelector("input,textarea");
      if (!input) {
        const fieldName = tr.dataset["fieldName"];
        tr.classList.add("-rgthree-editing");
        const isTextarea = fieldName === "userNote";
        const rowInput = $el(`${isTextarea ? "textarea" : 'input[type="text"]'}`, { value: td.textContent });
        rowInput.addEventListener("keydown", (evt) => {
          if (!isTextarea && evt.key === "Enter") {
            const modified = saveEditableRow(info, tr, true);
            this.modifiedModelData = this.modifiedModelData || modified;
            evt.stopPropagation();
            evt.preventDefault();
          } else if (evt.key === "Escape") {
            const modified = saveEditableRow(info, tr, false);
            this.modifiedModelData = this.modifiedModelData || modified;
            evt.stopPropagation();
            evt.preventDefault();
          }
        });
        appendChildren(empty(td), [rowInput]);
        rowInput.focus();
      } else if (target.nodeName.toLowerCase() === "button") {
        const modified = saveEditableRow(info, tr, true);
        this.modifiedModelData = this.modifiedModelData || modified;
      }
      e?.preventDefault();
      e?.stopPropagation();
    }
  }

  getInfoContent() {
    const info = this.modelInfo || {};
    const civitaiLink = info.links?.find((i) => i.includes("civitai.com/models"));
    const html = `
      <ul class="rgthree-info-area">
        <li title="Type" class="rgthree-info-tag -type -type-${(info.type || "").toLowerCase()}"><span>${info.type || ""}</span></li>
        <li title="Base Model" class="rgthree-info-tag -basemodel -basemodel-${(info.baseModel || "").toLowerCase()}"><span>${info.baseModel || ""}</span></li>
        <li class="rgthree-info-menu" stub="menu"></li>
      </ul>

      <table class="rgthree-info-table">
        ${infoTableRow("File", info.file || "")}
        ${infoTableRow("Hash (sha256)", info.sha256 || "")}
        ${
          civitaiLink
            ? infoTableRow("Civitai", `<a href="${civitaiLink}" target="_blank">${logoCivitai}View on Civitai</a>`)
            : info.raw?.civitai?.error === "Model not found"
              ? infoTableRow("Civitai", `<i>Model not found</i>`)
              : info.raw?.civitai?.error
                ? infoTableRow("Civitai", info.raw?.civitai?.error)
                : !info.raw?.civitai
                  ? infoTableRow("Civitai", `<button class="rgthree-button" data-action="fetch-civitai">Fetch info from civitai</button>`)
                  : ""
        }
        ${infoTableRow("Name", info.name || info.raw?.metadata?.ss_output_name || "", "Display name.", "name")}
        ${
          !info.baseModel && !info.baseModelFile
            ? ""
            : infoTableRow("Base Model", (info.baseModel || "") + (info.baseModelFile ? ` (${info.baseModelFile})` : ""))
        }
        ${!info.trainedWords?.length ? "" : infoTableRow("Trained Words", getTrainedWordsMarkup(info.trainedWords) ?? "", "Click to select for copy.")}
        ${
          !info.raw?.metadata?.ss_clip_skip || info.raw?.metadata?.ss_clip_skip == "None"
            ? ""
            : infoTableRow("Clip Skip", info.raw?.metadata?.ss_clip_skip)
        }
        ${infoTableRow("Strength Min", info.strengthMin ?? "", "Recommended minimum strength.", "strengthMin")}
        ${infoTableRow("Strength Max", info.strengthMax ?? "", "Recommended maximum strength.", "strengthMax")}
        ${infoTableRow("Additional Notes", info.userNote ?? "", "Local note.", "userNote")}
      </table>

      <ul class="rgthree-info-images">${
        info.images?.map(
          (img) => `
        <li>
          <figure>${
            img.type === "video" ? `<video src="${img.url}" autoplay loop></video>` : `<img src="${img.url}" />`
          }
            <figcaption>${imgInfoField("", img.civitaiUrl ? `<a href="${img.civitaiUrl}" target="_blank">civitai${link}</a>` : undefined)}${imgInfoField("seed", img.seed)}${imgInfoField("steps", img.steps)}${imgInfoField("cfg", img.cfg)}${imgInfoField("sampler", img.sampler)}${imgInfoField("model", img.model)}${imgInfoField("positive", img.positive)}${imgInfoField("negative", img.negative)}</figcaption>
          </figure>
        </li>`,
        ).join("") ?? ""
      }</ul>
    `;

    const div = $el("div", { html });

    setAttributes(query('[stub="menu"]', div), {
      children: [
        new MenuButton({
          icon: dotdotdot,
          options: [
            { label: "More Actions", type: "title" },
            {
              label: "Open API JSON",
              callback: async () => {
                if (this.modelInfo?.file) {
                  window.open(`/loraloader-mxd/api/loras/info?file=${encodeURIComponent(this.modelInfo.file)}`);
                }
              },
            },
            {
              label: "Clear all local info",
              callback: async () => {
                if (this.modelInfo?.file) {
                  this.modelInfo = await LORA_INFO_SERVICE.clearFetchedInfo(this.modelInfo.file);
                  this.setContent(this.getInfoContent());
                  this.setTitle(this.modelInfo?.name || this.modelInfo?.file || "Unknown");
                }
              },
            },
          ],
        }),
      ],
    });

    return div;
  }
}

export class MxdLoraInfoDialog extends MxdInfoDialog {
  async getModelInfo(file) {
    return LORA_INFO_SERVICE.getInfo(file, false, false);
  }

  async refreshModelInfo(file) {
    return LORA_INFO_SERVICE.refreshInfo(file);
  }
}

export class MxdCheckpointInfoDialog extends MxdInfoDialog {
  async getModelInfo(file) {
    return CHECKPOINT_INFO_SERVICE.getInfo(file, false, false);
  }

  async refreshModelInfo(file) {
    return CHECKPOINT_INFO_SERVICE.refreshInfo(file);
  }
}

function infoTableRow(name, value, help = "", editableFieldName = "") {
  return `
    <tr class="${editableFieldName ? "editable" : ""}" ${editableFieldName ? `data-field-name="${editableFieldName}"` : ""}>
      <td><span>${name} ${help ? `<span class="-help" title="${help}"></span>` : ""}<span></td>
      <td ${editableFieldName ? "" : 'colspan="2"'}>${String(value).startsWith("<") ? value : `<span>${value}<span>`}</td>
      ${editableFieldName ? `<td style="width: 24px;"><button class="rgthree-button-reset rgthree-button-edit" data-action="edit-row">${pencilColored}${diskColored}</button></td>` : ""}
    </tr>`;
}

function getTrainedWordsMarkup(words) {
  let markup = `<ul class="rgthree-info-trained-words-list">`;
  for (const wordData of words || []) {
    markup += `<li title="${wordData.word}" data-word="${wordData.word}" class="rgthree-info-trained-words-list-item" data-action="toggle-trained-word">
      <span>${wordData.word}</span>
      ${wordData.civitai ? logoCivitai : ""}
      ${wordData.count != null ? `<small>${wordData.count}</small>` : ""}
    </li>`;
  }
  markup += `</ul>`;
  return markup;
}

function saveEditableRow(info, tr, saving = true) {
  const fieldName = tr.dataset["fieldName"];
  const input = query("input,textarea", tr);
  let newValue = info[fieldName] ?? "";
  let modified = false;
  if (saving) {
    newValue = input.value;
    if (fieldName.startsWith("strength")) {
      if (Number.isNaN(Number(newValue))) {
        alert(`You must enter a number into the ${fieldName} field.`);
        return false;
      }
      newValue = (Math.round(Number(newValue) * 100) / 100).toFixed(2);
    }
    LORA_INFO_SERVICE.savePartialInfo(info.file, { [fieldName]: newValue });
    modified = true;
  }
  tr.classList.remove("-rgthree-editing");
  const td = query("td:nth-child(2)", tr);
  appendChildren(empty(td), [$el("span", { text: newValue })]);
  return modified;
}

function imgInfoField(label, value) {
  return value != null ? `<span>${label ? `<label>${label} </label>` : ""}${value}</span>` : "";
}




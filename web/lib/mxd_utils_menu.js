import { app } from "../../../scripts/app.js";
import { mxdApi } from "./mxd_api.js";
import { isSmartMatch } from "./mxd_smart_search.js";

let chooserStylesInjected = false;

function splitPathSegments(value) {
  return String(value)
    .split(/[\\/]+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

function ensureChooserTreeStyles() {
  if (chooserStylesInjected) return;
  chooserStylesInjected = true;

  const style = document.createElement("style");
  style.textContent = `
    .mxd-chooser-folder { opacity: 0.8; }
    .mxd-chooser-folder-arrow { display: inline-block; width: 15px; }
    .mxd-chooser-folder:hover { background-color: rgba(255, 255, 255, 0.1); }
    .mxd-chooser-prefix { display: none; }

    .litecontextmenu:has(input:not(:placeholder-shown)) .mxd-chooser-folder-contents {
      display: block !important;
    }
    .litecontextmenu:has(input:not(:placeholder-shown)) .mxd-chooser-folder {
      display: none;
    }
    .litecontextmenu:has(input:not(:placeholder-shown)) .mxd-chooser-prefix {
      display: inline;
    }
    .litecontextmenu:has(input:not(:placeholder-shown)) .litemenu-entry {
      padding-left: 2px !important;
    }
  `;
  document.body.appendChild(style);
}

function getContextMenuRootElement(menuInstance) {
  const menus = [...document.querySelectorAll(".litecontextmenu")];
  return (
    menuInstance?.root ||
    menuInstance?.rootElement ||
    menuInstance?.element ||
    menus[menus.length - 1] ||
    null
  );
}

function decorateChooserAsTree(menuEl) {
  if (!menuEl || menuEl.dataset.mxdChooserTreeApplied === "1") {
    return;
  }
  menuEl.dataset.mxdChooserTreeApplied = "1";

  const items = [...menuEl.querySelectorAll(".litemenu-entry")];
  if (!items.length) {
    return;
  }

  const folderMap = new Map();
  const rootItems = [];
  const itemsSymbol = Symbol("items");

  for (const item of items) {
    const dataValue = item.getAttribute("data-value") ?? item.textContent ?? "";
    const path = splitPathSegments(dataValue);
    if (!path.length) {
      continue;
    }

    item.textContent = path[path.length - 1];
    if (path.length > 1) {
      const prefix = document.createElement("span");
      prefix.className = "mxd-chooser-prefix";
      prefix.textContent = path.slice(0, -1).join("/") + "/";
      item.prepend(prefix);
    }

    if (path.length === 1) {
      rootItems.push(item);
      continue;
    }

    item.remove();

    let currentLevel = folderMap;
    for (let i = 0; i < path.length - 1; i++) {
      const folder = path[i];
      if (!currentLevel.has(folder)) {
        currentLevel.set(folder, new Map());
      }
      currentLevel = currentLevel.get(folder);
    }

    if (!currentLevel.has(itemsSymbol)) {
      currentLevel.set(itemsSymbol, []);
    }
    currentLevel.get(itemsSymbol).push(item);
  }

  const createFolderElement = (name) => {
    const folder = document.createElement("div");
    folder.className = "litemenu-entry mxd-chooser-folder";
    folder.style.paddingLeft = "5px";
    folder.innerHTML = `<span class="mxd-chooser-folder-arrow">></span> ${name}`;
    return folder;
  };

  const insertFolderStructure = (parentElement, map, level = 0) => {
    for (const [folderName, content] of map.entries()) {
      if (folderName === itemsSymbol) continue;

      const folderElement = createFolderElement(folderName);
      folderElement.style.paddingLeft = `${level * 10 + 5}px`;
      parentElement.appendChild(folderElement);

      const childContainer = document.createElement("div");
      childContainer.className = "mxd-chooser-folder-contents";
      childContainer.style.display = "none";

      const childItems = content.get(itemsSymbol) || [];
      for (const item of childItems) {
        item.style.paddingLeft = `${(level + 1) * 10 + 14}px`;
        childContainer.appendChild(item);
      }

      insertFolderStructure(childContainer, content, level + 1);
      parentElement.appendChild(childContainer);

      folderElement.addEventListener("click", (e) => {
        e.stopPropagation();
        const arrow = folderElement.querySelector(".mxd-chooser-folder-arrow");
        const contents = folderElement.nextElementSibling;
        if (!contents) return;
        if (contents.style.display === "none") {
          contents.style.display = "block";
          if (arrow) arrow.textContent = "v";
        } else {
          contents.style.display = "none";
          if (arrow) arrow.textContent = ">";
        }
      });
    }
  };

  const parent = rootItems[0]?.parentElement || items[0]?.parentElement || menuEl;
  if (!parent) return;
  const folderFragment = document.createDocumentFragment();
  insertFolderStructure(folderFragment, folderMap);
  parent.insertBefore(folderFragment, rootItems[0] || null);
}

// The built-in "Filter list" input (added by ComfyUI's own SearchFilter extension whenever
// a "dark" ContextMenu has >4 items) only does a literal lowercase substring match. We attach
// our own "input" listener *after* it so ours runs second and its display decision wins,
// re-scoring each item with token/typo-tolerant matching instead of requiring an exact substring.
function attachSmartChooserFilter(menuEl) {
  if (!menuEl) return;
  const filter = menuEl.querySelector(".comfy-context-menu-filter");
  if (!filter || filter.dataset.mxdSmartFilter === "1") return;
  filter.dataset.mxdSmartFilter = "1";

  filter.addEventListener("input", () => {
    const term = filter.value.trim();
    if (!term) return;

    const items = menuEl.querySelectorAll(".litemenu-entry");
    for (const item of items) {
      if (item.classList.contains("mxd-chooser-folder")) continue;
      const candidate = item.getAttribute("data-value") || item.textContent || "";
      item.style.display = isSmartMatch(term, candidate) ? "block" : "none";
    }
  });
}

function normalizeMenuCallbackValue(value) {
  if (typeof value === "object" && value && "content" in value) {
    return value.content;
  }
  return value;
}

/** Generic tree/search-enabled item chooser context menu, used for LoRAs, checkpoints, and
 * the smart UNET/CLIP model lists. */
export async function showModelChooser(event, callback, parentMenu, items, title = "Choose an Item") {
  const canvas = app.canvas;

  ensureChooserTreeStyles();

  const menu = new LiteGraph.ContextMenu(items, {
    event,
    parentMenu: parentMenu != null ? parentMenu : undefined,
    title,
    scale: Math.max(1, canvas.ds?.scale ?? 1),
    className: "dark",
    callback: (value, ...rest) => callback?.(normalizeMenuCallbackValue(value), ...rest),
  });

  requestAnimationFrame(() => {
    const menuEl = getContextMenuRootElement(menu);
    decorateChooserAsTree(menuEl);
    attachSmartChooserFilter(menuEl);
  });
}

export async function showLoraChooser(event, callback, parentMenu, loras) {
  if (!loras) {
    loras = await mxdApi.getLoras().then((items) => items.map((l) => l.file));
  }
  return showModelChooser(event, callback, parentMenu, loras, "Choose a LoRA");
}

// Folder-tree / grid display for the MXD latent loader combos (adapted from
// pysssss ComfyUI-Custom-Scripts' betterCombos; scoped to MXD nodes only).
import { app } from "../../../scripts/app.js";
import { $el } from "../../../scripts/ui.js";
import { api } from "../../../scripts/api.js";

const LOAD_LATENT_I2V_MXD = "LoadLatent_I2V_MXD";
const LOAD_LATENTS_FROM_FOLDER_I2V_MXD = "LoadLatents_FromFolder_I2V_MXD";
const LOAD_LATENT_I2V_PIPE_MXD = "LoadLatent_I2V_Pipe_MXD";
const LOAD_LATENTS_FROM_FOLDER_I2V_PIPE_MXD = "LoadLatents_FromFolder_I2V_Pipe_MXD";
const IMAGE_WIDTH = 384;
const IMAGE_HEIGHT = 384;

const NODE_CONFIGS = {
        [LOAD_LATENT_I2V_MXD]: { type: "latents", widgetName: "latent", hasImages: false },
        [LOAD_LATENTS_FROM_FOLDER_I2V_MXD]: { type: "latents", widgetName: "subfolder", hasImages: false },
        [LOAD_LATENT_I2V_PIPE_MXD]: { type: "latents", widgetName: "latent", hasImages: false },
        [LOAD_LATENTS_FROM_FOLDER_I2V_PIPE_MXD]: { type: "latents", widgetName: "subfolder", hasImages: false },
};

const CONFIG_BY_TYPE = Object.fromEntries(
        Object.values(NODE_CONFIGS).map((config) => [config.type, config])
);

const getNodeConfig = (nodeOrClass) => {
        const comfyClass = typeof nodeOrClass === "string" ? nodeOrClass : nodeOrClass?.comfyClass;
        return comfyClass ? NODE_CONFIGS[comfyClass] : undefined;
};

function encodeRFC3986URIComponent(str) {
        return encodeURIComponent(str).replace(/[!'()*]/g, (c) => `%${c.charCodeAt(0).toString(16).toUpperCase()}`);
}

const calculateImagePosition = (el, bodyRect) => {
	let { top, left, right } = el.getBoundingClientRect();
	const { width: bodyWidth, height: bodyHeight } = bodyRect;

	const isSpaceRight = right + IMAGE_WIDTH <= bodyWidth;
	if (isSpaceRight) {
		left = right;
	} else {
		left -= IMAGE_WIDTH;
	}

	top = top - IMAGE_HEIGHT / 2;
	if (top + IMAGE_HEIGHT > bodyHeight) {
		top = bodyHeight - IMAGE_HEIGHT;
	}
	if (top < 0) {
		top = 0;
	}

	return { left: Math.round(left), top: Math.round(top), isLeft: !isSpaceRight };
};

function showImage(relativeToEl, imageEl) {
	const bodyRect = document.body.getBoundingClientRect();
	if (!bodyRect) return;

	const { left, top, isLeft } = calculateImagePosition(relativeToEl, bodyRect);

	imageEl.style.left = `${left}px`;
	imageEl.style.top = `${top}px`;

	if (isLeft) {
		imageEl.classList.add("left");
	} else {
		imageEl.classList.remove("left");
	}

	document.body.appendChild(imageEl);
}

const imagesByType = {};
const imagePromises = {};

const loadImageList = async (type) => {
        imagesByType[type] = await (await api.fetchApi(`/pysssss/images/${type}`)).json();
        return imagesByType[type];
};

const ensureImageList = (type) => {
        const config = CONFIG_BY_TYPE[type];
        if (!config?.hasImages) {
                return Promise.resolve();
        }
        if (!imagePromises[type]) {
                imagePromises[type] = loadImageList(type);
        }
        return imagePromises[type];
};

for (const type of Object.keys(CONFIG_BY_TYPE)) {
        imagesByType[type] = {};
}

app.registerExtension({
        name: "mxd.Combo++",
        init() {
		const displayOptions = { "List (normal)": 0, "Tree (subfolders)": 1, "Thumbnails (grid)": 2 };
                const displaySetting = app.ui.settings.addSetting({
                        id: "mxd.Combo++.Submenu",
                        name: "Loader display mode (MXD Latent)",
			defaultValue: 1,
			type: "combo",
			options: (value) => {
				value = +value;

				return Object.entries(displayOptions).map(([k, v]) => ({
					value: v,
					text: k,
					selected: k === value,
				}));
			},
		});

		$el("style", {
			textContent: `
				.pysssss-combo-image {
					position: absolute;
					left: 0;
					top: 0;
					width: ${IMAGE_WIDTH}px;
					height: ${IMAGE_HEIGHT}px;
					object-fit: contain;
					object-position: top left;
					z-index: 9999;
				}
				.pysssss-combo-image.left {
					object-position: top right;
				}
				.pysssss-combo-folder { opacity: 0.7 }
				.pysssss-combo-folder-arrow { display: inline-block; width: 15px; }
				.pysssss-combo-folder:hover { background-color: rgba(255, 255, 255, 0.1); }
				.pysssss-combo-prefix { display: none }

				/* Special handling for when the filter input is populated to revert to normal */
				.litecontextmenu:has(input:not(:placeholder-shown)) .pysssss-combo-folder-contents {
					display: block !important;
				}
				.litecontextmenu:has(input:not(:placeholder-shown)) .pysssss-combo-folder { 
					display: none;
				}
				.litecontextmenu:has(input:not(:placeholder-shown)) .pysssss-combo-prefix { 
					display: inline;
				}
				.litecontextmenu:has(input:not(:placeholder-shown)) .litemenu-entry { 
					padding-left: 2px !important;
				}

				/* Grid mode */
				.pysssss-combo-grid {
					display: grid;
					grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
					gap: 10px;
					overflow-x: hidden;
					max-width: 60vw;
				}
				.pysssss-combo-grid .comfy-context-menu-filter {
					grid-column: 1 / -1;
					position: sticky;
					top: 0;
				}
				.pysssss-combo-grid .litemenu-entry {
					word-break: break-word;
					display: flex;
					flex-direction: column;
					justify-content: space-between;
					align-items: center;
				}
				.pysssss-combo-grid .litemenu-entry:before {
					content: "";
					display: block;
					width: 100%;
					height: 250px;
					background-size: contain;
					background-position: center;
					background-repeat: no-repeat;
					/* No-image image attribution: Picture icons created by Pixel perfect - Flaticon */
					background-image: var(--background-image, url(extensions/ComfyUI-Custom-Scripts/js/assets/no-image.png));
				}

			`,
			parent: document.body,
		});
                const imageHost = $el("img.pysssss-combo-image");

		const positionMenu = (menu, fillWidth) => {
			// compute best position
			let left = app.canvas.last_mouse[0] - 10;
			let top = app.canvas.last_mouse[1] - 10;

			const body_rect = document.body.getBoundingClientRect();
			const root_rect = menu.getBoundingClientRect();

			if (body_rect.width && left > body_rect.width - root_rect.width - 10) left = body_rect.width - root_rect.width - 10;
			if (body_rect.height && top > body_rect.height - root_rect.height - 10) top = body_rect.height - root_rect.height - 10;

			menu.style.left = `${left}px`;
			menu.style.top = `${top}px`;
			if (fillWidth) {
				menu.style.right = "10px";
			}
		};

                const updateMenu = async (menu, config) => {
                        const { type } = config;
                        try {
                                await ensureImageList(type);
                        } catch (error) {
                                console.error(error);
                                console.error("Error loading pysssss.betterCombos data");
                        }

			// Clamp max height so it doesn't overflow the screen
			const position = menu.getBoundingClientRect();
			const maxHeight = window.innerHeight - position.top - 20;
			menu.style.maxHeight = `${maxHeight}px`;

                        const images = imagesByType[type] || {};
			const items = menu.querySelectorAll(".litemenu-entry");

			// Add image handler to items
			const addImageHandler = (item) => {
				const text = item.getAttribute("data-value").trim();
				if (images[text]) {
					const textNode = document.createTextNode("*");
					item.appendChild(textNode);

					item.addEventListener(
						"mouseover",
						() => {
							imageHost.src = `/pysssss/view/${encodeRFC3986URIComponent(images[text])}?${+new Date()}`;
							document.body.appendChild(imageHost);
							showImage(item, imageHost);
						},
						{ passive: true }
					);
					item.addEventListener(
						"mouseout",
						() => {
							imageHost.remove();
						},
						{ passive: true }
					);
					item.addEventListener(
						"click",
						() => {
							imageHost.remove();
						},
						{ passive: true }
					);
				}
			};

			const createTree = () => {
				// Create a map to store folder structures
				const folderMap = new Map();
				const rootItems = [];
				const splitBy = (navigator.platform || navigator.userAgent).includes("Win") ? /\/|\\/ : /\//;
				const itemsSymbol = Symbol("items");

				// First pass - organize items into folder structure
                                for (const item of items) {
                                        let path = item
                                                .getAttribute("data-value")
                                                .split(splitBy)
                                                .filter((segment) => segment.length);

                                        if (type === "latents") {
                                                const latentsIndex = path.findIndex(
                                                        (segment) => segment.toLowerCase() === "latents"
                                                );
                                                if (latentsIndex !== -1) {
                                                        path = path.slice(latentsIndex + 1);
                                                }
                                        }

                                        if (!path.length) {
                                                item.remove();
                                                continue;
                                        }

                                        // Remove path from visible text
                                        item.textContent = path[path.length - 1];
					if (path.length > 1) {
						// Add the prefix path back in so it can be filtered on
						const prefix = $el("span.pysssss-combo-prefix", {
							textContent: path.slice(0, -1).join("/") + "/",
						});
						item.prepend(prefix);
					}

                                        addImageHandler(item);

					if (path.length === 1) {
						rootItems.push(item);
						continue;
					}

					// Temporarily remove the item from current position
					item.remove();

					// Create folder hierarchy
					let currentLevel = folderMap;
					for (let i = 0; i < path.length - 1; i++) {
						const folder = path[i];
						if (!currentLevel.has(folder)) {
							currentLevel.set(folder, new Map());
						}
						currentLevel = currentLevel.get(folder);
					}

					// Store the actual item in the deepest folder
					if (!currentLevel.has(itemsSymbol)) {
						currentLevel.set(itemsSymbol, []);
					}
					currentLevel.get(itemsSymbol).push(item);
				}

				const createFolderElement = (name) => {
					const folder = $el("div.litemenu-entry.pysssss-combo-folder", {
						innerHTML: `<span class="pysssss-combo-folder-arrow">></span> ${name}`,
						style: { paddingLeft: "5px" },
					});
					return folder;
				};

				const insertFolderStructure = (parentElement, map, level = 0) => {
					for (const [folderName, content] of map.entries()) {
						if (folderName === itemsSymbol) continue;

						const folderElement = createFolderElement(folderName);
						folderElement.style.paddingLeft = `${level * 10 + 5}px`;
						parentElement.appendChild(folderElement);

						const childContainer = $el("div.pysssss-combo-folder-contents", {
							style: { display: "none" },
						});

						// Add items in this folder
						const items = content.get(itemsSymbol) || [];
						for (const item of items) {
							item.style.paddingLeft = `${(level + 1) * 10 + 14}px`;
							childContainer.appendChild(item);
						}

						// Recursively add subfolders
						insertFolderStructure(childContainer, content, level + 1);
						parentElement.appendChild(childContainer);

						// Add click handler for folder
						folderElement.addEventListener("click", (e) => {
							e.stopPropagation();
							const arrow = folderElement.querySelector(".pysssss-combo-folder-arrow");
							const contents = folderElement.nextElementSibling;
							if (contents.style.display === "none") {
								contents.style.display = "block";
								arrow.textContent = "v";
							} else {
								contents.style.display = "none";
								arrow.textContent = ">";
							}
						});
					}
				};

				insertFolderStructure(items[0]?.parentElement || menu, folderMap);
				positionMenu(menu);
			};

			const addImageData = (item) => {
				const text = item.getAttribute("data-value").trim();
				if (images[text]) {
					item.style.setProperty("--background-image", `url(/pysssss/view/${encodeRFC3986URIComponent(images[text])})`);
				}
			};

			if (displaySetting.value === 1 || displaySetting.value === true) {
				createTree();
			} else if (displaySetting.value === 2) {
				menu.classList.add("pysssss-combo-grid");

				for (const item of items) {
					addImageData(item);
				}
				positionMenu(menu, true);
			} else {
                                for (const item of items) {
                                        addImageHandler(item);
                                }
                        }
                };

                const mutationObserver = new MutationObserver((mutations) => {
                        const node = app.canvas.current_node;

                        const config = getNodeConfig(node);
                        if (!node || !config) {
                                return;
                        }

			for (const mutation of mutations) {
				for (const removed of mutation.removedNodes) {
					if (removed.classList?.contains("litecontextmenu")) {
						imageHost.remove();
					}
				}

				for (const added of mutation.addedNodes) {
					if (added.classList?.contains("litecontextmenu")) {
						const overWidget = app.canvas.getWidgetAtCursor();
                                                if (overWidget?.name === config.widgetName) {
                                                        requestAnimationFrame(() => {
                                                                // Bad hack to prevent showing on right click menu by checking for the filter input
                                                                if (!added.querySelector(".comfy-context-menu-filter")) return;
                                                                updateMenu(added, config);
                                                        });
                                                }
                                                return;
					}
				}
			}
		});
		mutationObserver.observe(document.body, { childList: true, subtree: false });
	},
});

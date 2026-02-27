import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET_NODE_CLASS = "MXDCharacterEditDelete";
const TARGET_PROMPT_ENCODE_CLASS = "MXDCharacterPromptEncode";
const TARGET_WIDGET_NAME = "selected_character";
const NO_CHARACTERS_OPTION = "(no characters saved)";
const PICK_CHARACTER_OPTION = "(pick character)";

async function fetchCharacterNames() {
    try {
        const response = await api.fetchApi("/mxd/characters/list");
        if (!response.ok) {
            return [];
        }
        const data = await response.json();
        return Array.isArray(data) ? data : [];
    } catch {
        return [];
    }
}

function setComboOptions(widget, values) {
    const options = values.length ? values : [NO_CHARACTERS_OPTION];
    widget.options = widget.options || {};
    widget.options.values = options;

    if (!options.includes(widget.value)) {
        widget.value = options[0];
    }
}

function moveWidgetAfter(node, widgetToMove, referenceWidget) {
    const widgets = node.widgets || [];
    const moveIndex = widgets.indexOf(widgetToMove);
    const refIndex = widgets.indexOf(referenceWidget);
    if (moveIndex === -1 || refIndex === -1) {
        return;
    }

    widgets.splice(moveIndex, 1);

    const nextRefIndex = widgets.indexOf(referenceWidget);
    widgets.splice(nextRefIndex + 1, 0, widgetToMove);
}

function moveWidgetBefore(node, widgetToMove, referenceWidget) {
    const widgets = node.widgets || [];
    const moveIndex = widgets.indexOf(widgetToMove);
    const refIndex = widgets.indexOf(referenceWidget);
    if (moveIndex === -1 || refIndex === -1) {
        return;
    }
    widgets.splice(moveIndex, 1);
    const nextRefIndex = widgets.indexOf(referenceWidget);
    widgets.splice(nextRefIndex, 0, widgetToMove);
}

function installRefreshButton(node) {
    if (node.__mxdCharacterRefreshInstalled) {
        return;
    }
    const selectedWidget = (node.widgets || []).find((widget) => widget.name === TARGET_WIDGET_NAME);
    if (!selectedWidget) {
        return;
    }
    node.__mxdCharacterRefreshInstalled = true;

    const refresh = async () => {
        const names = await fetchCharacterNames();
        setComboOptions(selectedWidget, names);
        app.graph.setDirtyCanvas(true, true);
    };

    const refreshWidget = node.addWidget("button", "update list", null, refresh, {});
    refreshWidget.serialize = false;
    moveWidgetAfter(node, refreshWidget, selectedWidget);

    refresh();
}

function appendCharacterToPrompt(node, characterName) {
    if (!characterName || characterName === PICK_CHARACTER_OPTION || characterName === NO_CHARACTERS_OPTION) {
        return;
    }

    const promptWidget = (node.widgets || []).find((widget) => widget.name === "prompt");
    if (!promptWidget) {
        return;
    }

    const currentPrompt = typeof promptWidget.value === "string" ? promptWidget.value : "";
    const needsLeadingSpace = currentPrompt.length > 0 && !/\s$/.test(currentPrompt);
    const prefix = needsLeadingSpace ? " " : "";
    const suffix = /\s$/.test(characterName) ? "" : " ";
    promptWidget.value = `${currentPrompt}${prefix}${characterName}${suffix}`;
    app.graph.setDirtyCanvas(true, true);
}

function installPromptCharacterPicker(node) {
    if (node.__mxdCharacterPickerInstalled) {
        return;
    }

    const promptWidget = (node.widgets || []).find((widget) => widget.name === "prompt");
    if (!promptWidget) {
        return;
    }
    node.__mxdCharacterPickerInstalled = true;

    const pickerWidget = node.addWidget(
        "combo",
        "character_list",
        PICK_CHARACTER_OPTION,
        (value) => {
            appendCharacterToPrompt(node, value);
            pickerWidget.value = PICK_CHARACTER_OPTION;
            app.graph.setDirtyCanvas(true, true);
        },
        { values: [PICK_CHARACTER_OPTION] }
    );
    pickerWidget.serialize = false;

    const refreshPicker = async () => {
        const names = await fetchCharacterNames();
        const options = names.length ? [PICK_CHARACTER_OPTION, ...names] : [PICK_CHARACTER_OPTION, NO_CHARACTERS_OPTION];
        pickerWidget.options = pickerWidget.options || {};
        pickerWidget.options.values = options;
        if (!options.includes(pickerWidget.value)) {
            pickerWidget.value = PICK_CHARACTER_OPTION;
        }
        app.graph.setDirtyCanvas(true, true);
    };

    const refreshWidget = node.addWidget("button", "update list", null, refreshPicker, {});
    refreshWidget.serialize = false;

    const handlingWidget = (node.widgets || []).find((widget) => widget.name === "character_handling");
    if (handlingWidget) {
        moveWidgetBefore(node, pickerWidget, handlingWidget);
        moveWidgetBefore(node, refreshWidget, handlingWidget);
    }

    refreshPicker();
}

app.registerExtension({
    name: "mxd.character_prompts.refresh",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== TARGET_NODE_CLASS && nodeData.name !== TARGET_PROMPT_ENCODE_CLASS) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            if (nodeData.name === TARGET_NODE_CLASS) {
                installRefreshButton(this);
            } else if (nodeData.name === TARGET_PROMPT_ENCODE_CLASS) {
                installPromptCharacterPicker(this);
            }
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            if (nodeData.name === TARGET_NODE_CLASS) {
                installRefreshButton(this);
            } else if (nodeData.name === TARGET_PROMPT_ENCODE_CLASS) {
                installPromptCharacterPicker(this);
            }
            return result;
        };
    },
});

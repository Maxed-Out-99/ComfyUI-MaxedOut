import { app } from "../../../scripts/app.js";
import nspell from "../vendor/spellcheck/nspell.min.js";

const TEXTAREA_SELECTOR = "textarea.comfy-multiline-input";
const DICTIONARY_BASE = new URL("../vendor/spellcheck/", import.meta.url);
const WORD_PATTERN = /[\p{L}\p{M}]+(?:['’\-][\p{L}\p{M}]+)*/gu;
const MAX_SUGGESTIONS = 8;

const MIRROR_STYLE_PROPERTIES = [
    "borderBottomStyle",
    "borderBottomWidth",
    "borderLeftStyle",
    "borderLeftWidth",
    "borderRightStyle",
    "borderRightWidth",
    "borderTopStyle",
    "borderTopWidth",
    "boxSizing",
    "direction",
    "fontFamily",
    "fontFeatureSettings",
    "fontKerning",
    "fontSize",
    "fontStretch",
    "fontStyle",
    "fontVariant",
    "fontWeight",
    "letterSpacing",
    "lineHeight",
    "overflowX",
    "overflowY",
    "paddingBottom",
    "paddingLeft",
    "paddingRight",
    "paddingTop",
    "tabSize",
    "textAlign",
    "textIndent",
    "textRendering",
    "textTransform",
    "wordSpacing",
];

let spellchecker = null;

function loadDictionaryFile(name) {
    return fetch(new URL(name, DICTIONARY_BASE)).then((response) => {
        if (!response.ok) {
            throw new Error(`Could not load ${name}: HTTP ${response.status}`);
        }
        return response.text();
    });
}

async function loadSpellchecker() {
    try {
        const [aff, dic] = await Promise.all([
            loadDictionaryFile("en.aff"),
            loadDictionaryFile("en.dic"),
        ]);
        spellchecker = nspell(aff, dic);
    } catch (error) {
        console.warn("[ComfyUI-MaxedOut] Offline spell suggestions unavailable.", error);
    }
}

function appendMeasuredText(mirror, value) {
    let cursor = 0;

    for (const match of value.matchAll(WORD_PATTERN)) {
        const start = match.index;
        const end = start + match[0].length;

        if (start > cursor) {
            mirror.append(document.createTextNode(value.slice(cursor, start)));
        }

        const word = document.createElement("span");
        word.dataset.start = String(start);
        word.dataset.end = String(end);
        word.textContent = match[0];
        mirror.append(word);
        cursor = end;
    }

    if (cursor < value.length) {
        mirror.append(document.createTextNode(value.slice(cursor)));
    }
}

function createTextareaMirror(textarea) {
    const textareaRect = textarea.getBoundingClientRect();
    const computed = getComputedStyle(textarea);
    const mirror = document.createElement("div");
    const layoutWidth = textarea.offsetWidth || textareaRect.width;
    const layoutHeight = textarea.offsetHeight || textareaRect.height;
    const scaleX = textareaRect.width / layoutWidth;
    const scaleY = textareaRect.height / layoutHeight;

    mirror.setAttribute("aria-hidden", "true");
    mirror.style.position = "fixed";
    mirror.style.left = `${textareaRect.left}px`;
    mirror.style.top = `${textareaRect.top}px`;
    mirror.style.width = `${layoutWidth}px`;
    mirror.style.height = `${layoutHeight}px`;
    mirror.style.pointerEvents = "none";
    mirror.style.transform = `scale(${scaleX}, ${scaleY})`;
    mirror.style.transformOrigin = "top left";
    mirror.style.visibility = "hidden";
    mirror.style.whiteSpace = "pre-wrap";
    mirror.style.overflowWrap = "break-word";
    mirror.style.wordBreak = "break-word";

    for (const property of MIRROR_STYLE_PROPERTIES) {
        mirror.style[property] = computed[property];
    }

    appendMeasuredText(mirror, textarea.value);
    document.body.append(mirror);
    mirror.scrollLeft = textarea.scrollLeft;
    mirror.scrollTop = textarea.scrollTop;
    return mirror;
}

function wordRangeAtPoint(textarea, clientX, clientY) {
    const mirror = createTextareaMirror(textarea);

    try {
        for (const word of mirror.querySelectorAll("span[data-start]")) {
            for (const rect of word.getClientRects()) {
                if (
                    clientX >= rect.left &&
                    clientX <= rect.right &&
                    clientY >= rect.top &&
                    clientY <= rect.bottom
                ) {
                    return {
                        start: Number(word.dataset.start),
                        end: Number(word.dataset.end),
                    };
                }
            }
        }
    } finally {
        mirror.remove();
    }

    return null;
}

function matchCase(source, suggestion) {
    if (source === source.toUpperCase()) {
        return suggestion.toUpperCase();
    }
    if (source[0] === source[0].toUpperCase()) {
        return suggestion[0].toUpperCase() + suggestion.slice(1);
    }
    return suggestion;
}

function replaceWord(textarea, range, suggestion) {
    textarea.focus();
    textarea.setRangeText(suggestion, range.start, range.end, "end");
    textarea.dispatchEvent(
        new InputEvent("input", {
            bubbles: true,
            inputType: "insertReplacementText",
            data: suggestion,
        }),
    );
}

function menuValue(value) {
    return typeof value === "object" && value && "content" in value
        ? value.content
        : value;
}

function showSuggestions(event, textarea, range, sourceWord, suggestions) {
    new LiteGraph.ContextMenu(suggestions, {
        event,
        title: `Spelling: ${sourceWord}`,
        className: "dark",
        callback: (value) => {
            const suggestion = menuValue(value);
            if (typeof suggestion === "string") {
                replaceWord(textarea, range, suggestion);
            }
        },
    });
}

function handleContextMenu(event) {
    const textarea = event.target instanceof Element
        ? event.target.closest(TEXTAREA_SELECTOR)
        : null;

    if (!(textarea instanceof HTMLTextAreaElement) || !textarea.spellcheck || !spellchecker) {
        return;
    }

    const range = wordRangeAtPoint(textarea, event.clientX, event.clientY);
    if (!range) {
        return;
    }

    const sourceWord = textarea.value.slice(range.start, range.end);
    if (sourceWord.length > 64 || spellchecker.correct(sourceWord)) {
        return;
    }

    const suggestions = spellchecker
        .suggest(sourceWord)
        .slice(0, MAX_SUGGESTIONS)
        .map((suggestion) => matchCase(sourceWord, suggestion));

    if (!suggestions.length) {
        return;
    }

    event.preventDefault();
    event.stopImmediatePropagation();
    showSuggestions(event, textarea, range, sourceWord, suggestions);
}

// This spellchecker ships in two places: standalone as Spell-Check-MXD, and
// bundled here. ComfyUI auto-loads every .js under a pack's web directory, so
// with both packs installed this file runs twice -- which would load two copies
// of the 550 KB dictionary and attach two capture-phase contextmenu listeners,
// making every right-click open the suggestion menu twice. Whichever copy
// reaches setup() first claims the global; the other stands down. The flag name
// is deliberately identical in both packs -- do not rename it in only one.
const SPELLCHECK_GUARD = "__mxdPromptSpellcheckActive";

app.registerExtension({
    name: "ComfyUI-MaxedOut.PromptSpellcheck",

    setup() {
        if (window[SPELLCHECK_GUARD]) {
            return;
        }
        window[SPELLCHECK_GUARD] = true;
        loadSpellchecker();
        document.addEventListener("contextmenu", handleContextMenu, true);
    },
});

import logging

import torch

from . import resolver, storage

try:
    from aiohttp import web
    from server import PromptServer

    HAVE_SERVER = True
except Exception:
    web = None
    PromptServer = None
    HAVE_SERVER = False


CHARACTER_LIST_ROUTE = "/mxd/characters/list"
NO_CHARACTERS_OPTION = "(no characters saved)"


def _list_character_names() -> list[str]:
    schema = storage.load_schema()
    return storage.list_character_names(schema)


def _selected_character_input():
    names = _list_character_names()
    options = names if names else [NO_CHARACTERS_OPTION]
    return (
        options,
        {
            "default": options[0],
            "tooltip": "Choose a saved character. Use the refresh button to update this list.",
        },
    )


def _is_valid_selected(name: str) -> bool:
    return bool(name and name != NO_CHARACTERS_OPTION)


if HAVE_SERVER:
    routes = PromptServer.instance.routes

    @routes.get(CHARACTER_LIST_ROUTE)
    async def mxd_character_prompt_list(_request):
        names = _list_character_names()
        if not names:
            names = [NO_CHARACTERS_OPTION]
        return web.json_response(names)


def _encode_text(clip, text: str):
    tokens = clip.tokenize(text)
    return clip.encode_from_tokens_scheduled(tokens)


def _conditioning_concat(conditioning_to, conditioning_from):
    out = []
    if len(conditioning_from) > 1:
        logging.warning(
            "MXD Character Prompt Encode: concat source contains more than one condition; using first condition only."
        )

    cond_from = conditioning_from[0][0]
    for i in range(len(conditioning_to)):
        cond_to = conditioning_to[i][0]
        merged = torch.cat((cond_to, cond_from), 1)
        out.append([merged, conditioning_to[i][1].copy()])
    return out


def _conditioning_combine(conditioning_1, conditioning_2):
    return conditioning_1 + conditioning_2


def _encode_segments(clip, segments: list[str], break_mode: str):
    if not segments:
        return _encode_text(clip, "")
    conditioning = _encode_text(clip, segments[0])
    for segment in segments[1:]:
        segment_cond = _encode_text(clip, segment)
        if break_mode == "concat_characters_then_scene":
            conditioning = _conditioning_concat(conditioning, segment_cond)
        elif break_mode == "combine_characters_then_scene":
            conditioning = _conditioning_combine(conditioning, segment_cond)
    return conditioning


class MXDCharacterCreate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character_name": ("STRING", {"default": "", "multiline": False}),
                "character_prompt": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "MXD/Character"
    TITLE = "Character Create"

    def run(self, character_name, character_prompt):
        try:
            canonical = storage.normalize_key(character_name)
            existing_schema = storage.load_schema()
            existed = canonical in existing_schema["characters"]
            schema, entry = storage.save_or_update_character(
                character_name=character_name,
                character_prompt=character_prompt,
                aliases_csv=None,
            )
            count = len(schema["characters"])
            action = "Updated" if existed else "Created"
            status = f"{action} character '{entry['name']}' ({count} total)."
            return (status,)
        except Exception as error:
            return (f"Error: {error}",)


class MXDCharacterEditDelete:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "selected_character": _selected_character_input(),
                "action": (
                    ["load_character", "edit_character", "delete_character"],
                    {"default": "load_character"},
                ),
                "character_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Used when action=edit_character.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("character_prompt",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "MXD/Character"
    TITLE = "Character Edit/Delete"

    def run(self, selected_character, action, character_prompt):
        try:
            if not _is_valid_selected(selected_character):
                return ("",)

            if action == "load_character":
                _schema, _canonical, entry = storage.load_character(selected_character)
                return (entry["prompt"],)

            if action == "edit_character":
                _, _canonical, current_entry = storage.load_character(selected_character)
                _schema, updated_entry = storage.save_or_update_character(
                    character_name=current_entry["name"],
                    character_prompt=character_prompt,
                    aliases_csv=None,
                )
                return (updated_entry["prompt"],)

            storage.delete_character(selected_character)
            return ("",)
        except Exception:
            return ("",)


class MXDCharacterPromptEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True}),
                "character_handling": (
                    ["separate_with_break", "combine_conditioning", "single_prompt"],
                    {"default": "separate_with_break"},
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "prompt_with_character_text")
    FUNCTION = "encode"
    CATEGORY = "MXD/Character"
    TITLE = "Character Prompt Encode"

    def encode(self, clip, prompt, character_handling):
        if clip is None:
            raise RuntimeError(
                "ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model."
            )

        schema = storage.load_schema()
        characters = schema["characters"]

        resolved = resolver.resolve_prompt(
            prompt=prompt,
            characters=characters,
            reference_mode="plain_only",
            missing_behavior="warn_and_keep_text",
        )

        used_prompts = []
        for canonical in resolved.used_canonical_names:
            entry = characters.get(canonical)
            if entry and entry.get("prompt"):
                used_prompts.append(entry["prompt"])

        if character_handling != "single_prompt" and len(used_prompts) >= 2:
            segments = used_prompts + [resolved.scene_prompt]
            if character_handling == "separate_with_break":
                break_mode = "concat_characters_then_scene"
            else:
                break_mode = "combine_characters_then_scene"
            conditioning = _encode_segments(clip, segments, break_mode)
        else:
            conditioning = _encode_text(clip, resolved.resolved_prompt)

        return (conditioning, resolved.resolved_prompt)


NODE_CLASS_MAPPINGS = {
    "MXDCharacterCreate": MXDCharacterCreate,
    "MXDCharacterEditDelete": MXDCharacterEditDelete,
    "MXDCharacterPromptEncode": MXDCharacterPromptEncode,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "MXDCharacterCreate": "Character Create MXD",
    "MXDCharacterEditDelete": "Character Edit/Delete MXD",
    "MXDCharacterPromptEncode": "Character Prompt Encode MXD",
}

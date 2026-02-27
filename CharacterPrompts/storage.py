import json
import os
from datetime import datetime, timezone

import folder_paths

SCHEMA_VERSION = 1
EXTENSION_DIRNAME = "ComfyUI-CharacterPrompts-MXD"
FILENAME = "characters.json"


def normalize_key(value: str) -> str:
    return " ".join(value.strip().lower().split())


def parse_aliases_csv(aliases_csv: str) -> list[str]:
    if not aliases_csv:
        return []
    aliases = []
    seen = set()
    for raw in aliases_csv.split(","):
        alias = normalize_key(raw)
        if alias and alias not in seen:
            aliases.append(alias)
            seen.add(alias)
    return aliases


def get_library_dir() -> str:
    return os.path.join(folder_paths.get_user_directory(), "default", EXTENSION_DIRNAME)


def get_library_path() -> str:
    return os.path.join(get_library_dir(), FILENAME)


def _empty_schema() -> dict:
    return {"version": SCHEMA_VERSION, "characters": {}}


def _write_json_atomic(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _normalize_entry(canonical: str, entry: dict) -> dict:
    name = str(entry.get("name", canonical)).strip() or canonical
    prompt = str(entry.get("prompt", "")).strip()
    aliases_raw = entry.get("aliases", [])
    aliases = []
    seen = {canonical}
    if isinstance(aliases_raw, list):
        for alias_raw in aliases_raw:
            alias = normalize_key(str(alias_raw))
            if alias and alias not in seen:
                aliases.append(alias)
                seen.add(alias)
    updated_at = str(entry.get("updated_at", _utc_now()))
    return {
        "name": name,
        "prompt": prompt,
        "aliases": aliases,
        "updated_at": updated_at,
    }


def _normalize_schema(raw: dict) -> dict:
    if not isinstance(raw, dict):
        return _empty_schema()
    characters_raw = raw.get("characters", {})
    if not isinstance(characters_raw, dict):
        characters_raw = {}
    normalized = {}
    for key, entry in characters_raw.items():
        canonical = normalize_key(str(key))
        if not canonical or not isinstance(entry, dict):
            continue
        normalized[canonical] = _normalize_entry(canonical, entry)
    return {"version": SCHEMA_VERSION, "characters": normalized}


def _backup_corrupt_file(path: str) -> None:
    if not os.path.exists(path):
        return
    backup_path = f"{path}.bak"
    try:
        if os.path.exists(backup_path):
            os.remove(backup_path)
        os.replace(path, backup_path)
    except OSError:
        pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_schema() -> dict:
    path = get_library_path()
    if not os.path.exists(path):
        schema = _empty_schema()
        _write_json_atomic(path, schema)
        return schema

    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError):
        _backup_corrupt_file(path)
        schema = _empty_schema()
        _write_json_atomic(path, schema)
        return schema

    schema = _normalize_schema(raw)
    if schema != raw:
        _write_json_atomic(path, schema)
    return schema


def save_schema(schema: dict) -> None:
    normalized = _normalize_schema(schema)
    _write_json_atomic(get_library_path(), normalized)


def list_character_names(schema: dict | None = None) -> list[str]:
    schema = schema or load_schema()
    names = [entry["name"] for entry in schema["characters"].values()]
    return sorted(names, key=lambda value: value.lower())


def save_or_update_character(
    character_name: str,
    character_prompt: str,
    aliases_csv: str | None = "",
) -> tuple[dict, dict]:
    canonical = normalize_key(character_name)
    if not canonical:
        raise ValueError("Character name cannot be empty.")

    prompt = str(character_prompt or "").strip()
    if not prompt:
        raise ValueError("Character prompt cannot be empty.")

    schema = load_schema()
    existing_entry = schema["characters"].get(canonical)

    if aliases_csv is None:
        aliases_raw = existing_entry.get("aliases", []) if existing_entry else []
        aliases = []
        seen = {canonical}
        for alias_raw in aliases_raw:
            alias = normalize_key(str(alias_raw))
            if alias and alias not in seen:
                aliases.append(alias)
                seen.add(alias)
    else:
        aliases = [alias for alias in parse_aliases_csv(aliases_csv) if alias != canonical]

    schema["characters"][canonical] = {
        "name": character_name.strip() or canonical,
        "prompt": prompt,
        "aliases": aliases,
        "updated_at": _utc_now(),
    }
    save_schema(schema)
    return schema, schema["characters"][canonical]


def load_character(character_name: str) -> tuple[dict, str, dict]:
    canonical = normalize_key(character_name)
    if not canonical:
        raise ValueError("Character name cannot be empty.")
    schema = load_schema()
    entry = schema["characters"].get(canonical)
    if entry is None:
        raise KeyError(f"Character '{character_name}' was not found.")
    return schema, canonical, entry


def delete_character(character_name: str) -> tuple[dict, str]:
    canonical = normalize_key(character_name)
    if not canonical:
        raise ValueError("Character name cannot be empty.")
    schema = load_schema()
    if canonical not in schema["characters"]:
        raise KeyError(f"Character '{character_name}' was not found.")
    removed_name = schema["characters"][canonical]["name"]
    del schema["characters"][canonical]
    save_schema(schema)
    return schema, removed_name

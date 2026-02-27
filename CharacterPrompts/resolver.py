import logging
import re
from dataclasses import dataclass

from .storage import normalize_key


@dataclass
class ResolveResult:
    resolved_prompt: str
    scene_prompt: str
    used_canonical_names: list[str]
    missing_tags: list[str]


def _build_lookup(characters: dict) -> dict[str, str]:
    lookup = {}
    for canonical, entry in characters.items():
        lookup[canonical] = canonical
        for alias in entry.get("aliases", []):
            alias_key = normalize_key(str(alias))
            if alias_key:
                lookup[alias_key] = canonical
    return lookup


def _add_used(used: list[str], canonical: str) -> None:
    if canonical not in used:
        used.append(canonical)


def _scene_prompt_from_original(prompt: str) -> str:
    # Keep scene wording intact, only drop explicit @ marker.
    return re.sub(r"@([A-Za-z0-9_\-]+)", r"\1", prompt)


def _find_missing_tags(prompt: str, lookup: dict[str, str]) -> list[str]:
    seen = set()
    missing = []
    for match in re.finditer(r"@([A-Za-z0-9_\-]+)", prompt):
        raw = match.group(1)
        key = normalize_key(raw)
        if key not in lookup and key not in seen:
            missing.append(raw)
            seen.add(key)
    return missing


def _pattern_from_terms(terms: list[str], prefix: str = "", suffix: str = "") -> re.Pattern | None:
    if not terms:
        return None
    escaped = [re.escape(term) for term in terms]
    pattern = f"{prefix}({'|'.join(escaped)}){suffix}"
    return re.compile(pattern, re.IGNORECASE)


def _replace_with_placeholders(
    text: str,
    pattern: re.Pattern | None,
    lookup: dict[str, str],
    characters: dict,
    used: list[str],
    placeholders: dict[str, str],
    next_id_start: int = 1,
) -> tuple[str, int]:
    if pattern is None:
        return text, next_id_start

    next_id = next_id_start

    def callback(match: re.Match) -> str:
        nonlocal next_id
        matched = normalize_key(match.group(1))
        canonical = lookup.get(matched)
        if canonical is None:
            return match.group(0)
        _add_used(used, canonical)
        placeholder = f"__MXD_CHAR_{next_id}__"
        next_id += 1
        placeholders[placeholder] = characters[canonical]["prompt"]
        return placeholder

    return pattern.sub(callback, text), next_id


def _handle_missing_tags(missing: list[str], missing_behavior: str) -> None:
    if not missing:
        return
    missing_text = ", ".join(missing)
    if missing_behavior == "error":
        raise ValueError(f"Unknown character tags: {missing_text}")
    if missing_behavior == "warn_and_keep_text":
        logging.warning("MXD Character Prompt Encode: unknown character tags left unchanged: %s", missing_text)


def resolve_prompt(
    prompt: str,
    characters: dict,
    reference_mode: str = "support_both",
    missing_behavior: str = "warn_and_keep_text",
) -> ResolveResult:
    lookup = _build_lookup(characters)
    used = []
    placeholders = {}
    text = prompt
    next_id = 1

    sorted_terms = sorted(lookup.keys(), key=len, reverse=True)
    tag_pattern = _pattern_from_terms(sorted_terms, prefix=r"(?<!\w)@", suffix=r"(?![\w\-])")
    plain_pattern = _pattern_from_terms(sorted_terms, prefix=r"(?<![@\w])", suffix=r"(?![\w\-])")

    if reference_mode in ("support_both", "tags_only"):
        text, next_id = _replace_with_placeholders(
            text=text,
            pattern=tag_pattern,
            lookup=lookup,
            characters=characters,
            used=used,
            placeholders=placeholders,
            next_id_start=next_id,
        )

    if reference_mode in ("support_both", "plain_only"):
        text, next_id = _replace_with_placeholders(
            text=text,
            pattern=plain_pattern,
            lookup=lookup,
            characters=characters,
            used=used,
            placeholders=placeholders,
            next_id_start=next_id,
        )

    for placeholder, replacement in placeholders.items():
        text = text.replace(placeholder, replacement)

    missing = _find_missing_tags(prompt, lookup) if reference_mode in ("support_both", "tags_only") else []
    _handle_missing_tags(missing, missing_behavior)

    return ResolveResult(
        resolved_prompt=text,
        scene_prompt=_scene_prompt_from_original(prompt),
        used_canonical_names=used,
        missing_tags=missing,
    )

# ComfyUI-CharacterPrompts-MXD

Beginner-friendly character alias nodes for ComfyUI.

## What This Adds

1. `Character Create`
- Create a new character or update an existing one by name.

2. `Character Edit/Delete`
- Pick from a dropdown of saved characters.
- Includes a refresh button on the dropdown.
- Load selected character data, save prompt changes, or delete selected.

3. `Character Prompt Encode`
- Resolves character names in your prompt (`Homayun`) and tags (`@Homayun`).
- Replaces references with the saved character description text.
- Outputs `CONDITIONING` directly for sampler wiring.
- Supports multi-character handling modes for cleaner multi-character prompts.

## Storage Location

Character library is shared globally at:

`<ComfyUI user>/default/ComfyUI-CharacterPrompts-MXD/characters.json`

Example:

`ComfyUI/user/default/ComfyUI-CharacterPrompts-MXD/characters.json`

## Quick Start

1. Add `Character Create`.
2. Fill:
- `character_name`: `Homayun`
- `character_prompt`: your full character description
3. Queue once to save.
4. Optional edits/deletes:
- Add `Character Edit/Delete`.
- Pick character from dropdown.
- Use refresh button if needed.
5. Add `Character Prompt Encode`.
6. Connect your `CLIP` input.
7. Write a normal scene prompt:
- `Homayun holding flowers`
- or `@Homayun holding flowers`
8. Use `conditioning` output as your positive conditioning.

## Recommended Settings

- `character_handling`: `separate_with_break`

## Notes

- When 2+ characters are detected and `character_handling` is `separate_with_break` or `combine_conditioning`, each character description is encoded as a separate segment before scene text.
- If fewer than 2 characters are detected, it falls back to a single resolved prompt encode.

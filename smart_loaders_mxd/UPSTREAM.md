# Upstream sync notes

`dequant.py`, `loader.py`, `ops.py`, and `tools/convert.py` in this package are
vendored **verbatim** from [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)
(Apache-2.0 — see `LICENSE` in this folder). `nodes.py` is our own adaptation
(unified UNET/CLIP loaders that pick between safetensors and GGUF automatically)
and is not meant to track upstream 1:1.

## Baseline

Vendored from `Maxed-Out-99/ComfyUI-SmartModelLoaders-MXD` at commit
`7fd86b156afbe3bb453958900d5b6bd8c074ab8a` (2026-02-22, "Add Gemma3 GGUF support
and mmap/memory fixes"). That repo is a manual copy of city96/ComfyUI-GGUF
rather than a git fork, but as of the last check (2026-06-13) its
`dequant.py`/`loader.py`/`ops.py` were byte-identical to city96's `main`
branch — so this baseline should be treated as "current with upstream `main`"
as of that date.

## How to sync when city96 ships an update

1. Download the current `dequant.py`, `loader.py`, `ops.py`, and
   `tools/convert.py` from
   `https://github.com/city96/ComfyUI-GGUF/tree/main`.
2. Diff each one against the matching file in this folder (ignore the header
   comment block added here). If the diff is clean (no local modifications
   to reconcile — these 4 files are never hand-edited), just drop the new
   versions in and re-add the vendoring header comment to the top of each.
3. Leave `nodes.py` alone — cross-check it against upstream's `nodes.py` only
   if you want to pull in a new *loader node* (e.g. their `GGUFOps`
   changes), and merge by hand since ours diverges intentionally (unified
   loaders, MXD info-icon UI hooks).
4. Bump the baseline note above (commit/date) after syncing.

## What was intentionally left out

The standalone repo's `tools/` folder also has `fix_5d_tensors.py`,
`fix_lines_ending.py`, `lcpp.patch`, and `read_tensors.py` — one-off CLI/dev
scripts, not runtime dependencies. Only `convert.py` was vendored here,
because `loader.py`'s compatibility-mode fallback does
`from .tools.convert import detect_arch` at runtime for GGUF files missing
standard architecture metadata.

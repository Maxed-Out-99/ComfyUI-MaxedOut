# ComfyUI-MaxedOut

Custom ComfyUI nodes I use in my own Maxed Out workflows (SDXL, Flux, WAN 2.2, and more).

![GitHub stars](https://img.shields.io/github/stars/Maxed-Out-99/ComfyUI-MaxedOut?style=flat-square)
![GitHub issues](https://img.shields.io/github/issues/Maxed-Out-99/ComfyUI-MaxedOut?style=flat-square)
![ComfyUI custom nodes](https://img.shields.io/badge/ComfyUI-Custom%20Nodes-1f6feb?style=flat-square)

<p align="left">
  <img src="assets/maxedout_icon.png" alt="Maxed Out icon" width="96" />
</p>

## Mission

My main goal is to make existing nodes easier to use in the day-to-day especially for my own workflows.

## Overview

- Cleaner UX on frequently used nodes.
- Time-saving presets for common resolutions.
- Image/Video Comparer nodes with easy save.
- Advanced Wan 2.2 nodes for my Patreon exclusive workflows.

## Install

ComfyUI Manager (Recommended):

Open Manager (top-right in ComfyUI), search `Maxed Out`, install, then restart ComfyUI.

Manually:

```powershell
cd ComfyUI\custom_nodes
git clone https://github.com/Maxed-Out-99/ComfyUI-MaxedOut.git
```

Restart ComfyUI after install.

## Model Storage Auto-Registration

If you keep models outside `ComfyUI/models` (e.g. on another drive), point MaxedOut at that folder and every subfolder inside it (`loras`, `checkpoints`, `vae`, ...) gets auto-registered with ComfyUI at startup, just like `ComfyUI/models/<type>`. No `extra_model_paths.yaml` edits needed, and any new subfolder you add just works after a restart.

Set it any one of these ways:

- **ComfyUI Settings panel** (easiest): Settings → `MXD` → `Model Storage` → set *Root Folder*, then restart ComfyUI.
- **Config file**: copy `model_storage_config.json.example` to `model_storage_config.json` in this repo's folder and set `model_storage_root`. This file is gitignored, so it's yours to keep.
- **Environment variable**: set `MAXEDOUT_MODEL_STORAGE` to the folder path before launching ComfyUI.

Leave all three unset and nothing changes -- this is entirely opt-in.

## Featured Nodes

| Node | What it does |
|---|---|
| `Lora Loader MXD` | Based on rgthree Power LoRA Loader. Fixes copy/paste issues, surfaces useful info/remove buttons, and keeps local LoRA info files organized. |
| `Image Comparer MXD` | Compare original vs new images in-node and save the new result easily. |
| `Video Comparer MXD` | Similar to Image Comparer but for video. |
| `Flux Empty Latent Image MXD` / `ZIT Empty Latent Image MXD` / `SDXL Empty Latent Image MXD` | Resolution presets plus vertical toggle to avoid retyping the same sizes repeatedly. |
| `Save Image MXD` | Simple save modes (`Save + Preview`, `Save Only`, `Preview Only`). |
| `WAN 2.2 MXD` nodes | Helpers for WAN 2.2 latent/video prep, frame tools, and I2V-focused workflows. |
| Prompt spellcheck | Right-click a misspelled word in any prompt box for suggestions. Works offline, no node to add. |

## Companion Packs

Some things that used to live in here now ship separately, so you can install
only what you want. Nothing below is required.

| Pack | What it is |
|---|---|
| [Live-Preview-MXD](https://github.com/Maxed-Out-99/Live-Preview-MXD) | Watch video generations animate while they render, in a dockable panel. Also adds LTX 2.3 previews, which core can't do at all. |
| [Prompt-Library-MXD](https://github.com/Maxed-Out-99/Prompt-Library-MXD) | Save named prompt snippets by category and pull them in by name, or let a wildcard pick one at random. |
| [Spell-Check-MXD](https://github.com/Maxed-Out-99/Spell-Check-MXD) | The prompt spellchecker on its own, if you don't want the rest of this pack. Safe to install alongside — it won't double up. |
| [Smart-Model-Loaders-MXD](https://github.com/Maxed-Out-99/Smart-Model-Loaders-MXD) | Loader nodes that accept safetensors or GGUF in the same slot. Only needed if you actually use GGUF quants. |

## Free Workflows

Free workflow releases are posted on Patreon (no sign up needed).

- [SDXL v1.5 Here](https://www.patreon.com/posts/free-sdxl-v1-5-129782694?utm_medium=clipboard_copy&utm_source=copyLink&utm_campaign=postshare_creator&utm_content=join_link)

![SDXL v1.5 Free Workflow](assets/sdxlfree.png)

- [Flux v1.5 Here](https://www.patreon.com/posts/free-flux-just-131945103?utm_medium=clipboard_copy&utm_source=copyLink&utm_campaign=postshare_creator&utm_content=join_link)

![Flux v1.5 Free Workflow](assets/fluxfree.png)

## Credits

Huge thanks to these projects. I have learned a lot from them and built on many of their ideas:

- https://github.com/rgthree/rgthree-comfy
  Inspiration for LoRA Loader, Image/Video Comparer, and more.
- https://github.com/kijai/ComfyUI-KJNodes
  Major reference and inspiration for my own nodes.

If you star this repo, definitely consider starring theirs too.

## License

MIT — see [LICENSE](LICENSE). Vendored code under `web/vendor/` keeps its own
licenses, carried alongside it in that folder.

This pack has no pip dependencies — everything it needs ships with ComfyUI.

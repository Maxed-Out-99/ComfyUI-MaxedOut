# ComfyUI-MaxedOut

Custom ComfyUI nodes built for production workflows, with a focus on SDXL, Flux, and WAN 2.2.

![GitHub stars](https://img.shields.io/github/stars/Maxed-Out-99/ComfyUI-MaxedOut?style=flat-square)
![GitHub issues](https://img.shields.io/github/issues/Maxed-Out-99/ComfyUI-MaxedOut?style=flat-square)
![ComfyUI custom nodes](https://img.shields.io/badge/ComfyUI-Custom%20Nodes-1f6feb?style=flat-square)

<p align="left">
  <img src="assets/maxedout_icon.png" alt="Maxed Out icon" width="96" />
</p>

## Why This Repo

These are the nodes I use in my own workflows. The goal is speed, cleaner graphs, and fewer repetitive setup steps.

## Install

```powershell
cd ComfyUI\custom_nodes
git clone https://github.com/Maxed-Out-99/ComfyUI-MaxedOut.git
```

Restart ComfyUI after install.

## Featured Nodes

| Node | What it does |
|---|---|
| `Lora Loader MXD` | Power LoRA loading workflow inspired by rgthree, with practical UX updates and local info tools. |
| `Image Comparer + Save MXD` | Compare original vs new images in-node and save the new result quickly. |
| `Video Comparer MXD` | Side-by-side style video comparison flow for before/after checks. |
| `Flux Empty Latent Image MXD` / `Flux 2 Empty Latent Image MXD` / `SDXL Empty Latent Image MXD` | Resolution presets so you do not keep retyping common sizes. |
| `Save Image MXD` | Simple save modes (`Save + Preview`, `Save Only`, `Preview Only`). |
| `WAN 2.2 MXD` nodes | Utility nodes for latent/video prep and I2V support in WAN 2.2 workflows. |

## Free Workflows

Free workflow releases are posted on Patreon:

- SDXL v1.5 (Free): `[add free Patreon link here]`
- Mega Flux v1.5 (Free): `[add free Patreon link here]`
- Patreon profile: `[add Patreon profile link here]`

## Showcase (Placeholders)

- [Put SDXL v1.5 image here]
- [Put Mega Flux v1.5 image here]
- [Put Image Comparer screenshot here]
- [Put Video Comparer preview GIF/video here]

## Notes

- `Lora Loader MXD` is integrated directly in this repo.
- Local LoRA info is stored in a structured `more_info` area to avoid cluttering LoRA folders.

## Credits

- `Lora Loader MXD` builds on ideas from rgthree Power LoRA Loader.
- Media comparer nodes are based on the rgthree comparer concept, with MXD-focused UX adjustments.

## Support

If these nodes help your workflow:

- Star this repo.
- Share a result using the nodes.
- Check the free workflows above.

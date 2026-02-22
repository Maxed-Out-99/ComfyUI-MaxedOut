# ComfyUI-MaxedOut
Custom ComfyUI nodes used in Maxed Out workflows (SDXL, Flux, Wan 2.2, etc.).

## Included Nodes
- Core MaxedOut nodes from `maxedoutnodes.py`
- Media comparer nodes from `mediacomparers.py`
- WAN 2.2 nodes from `wan22nodes.py`
- `Lora Loader MXD` (merged from `ComfyUI-LoraLoader-MXD`)

## Lora Loader MXD Merge
`Lora Loader MXD` is now integrated directly into this repository.

Compatibility and behavior retained:
- Node type remains `Lora Loader MXD`
- API namespace remains `/loraloader-mxd/api/...`
- Sidecar metadata compatibility remains `*.rgthree-info.json`
- Frontend extension name remains `mxd.PowerLoraLoader`

## Migration Notes
- Existing `ComfyUI-LoraLoader-MXD` installs are supported by a one-release compatibility shim.
- During transition, keep both repos installed if needed; the standalone shim defers to this merged provider to avoid duplicate registration.
- After the transition release window, the standalone repo can be removed.

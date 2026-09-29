# Third-party licenses

## ComfyUI-KJNodes and ComfyUI-VideoHelperSuite

The animated latent-preview implementation in `nodes/ltx/preview.py` and
`system/live_preview.py` includes logic adapted from
[ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) by Kijai and
[ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
by Kosinkadink. Both upstream projects are licensed under the GNU General
Public License, Version 3. The adapted code was modified by Maxed Out in 2026
for automatic LTX 2.3 fallback decoding, private preview events, persistent
preview output, and integration with ComfyUI-MaxedOut.

ComfyUI-MaxedOut is distributed under the GNU General Public License, Version
3. The complete license text is included in the repository-root `LICENSE`.

## ComfyUI-Krea2Edit

`nodes/krea2_edit_core.py` is adapted from
[ComfyUI-Krea2Edit](https://github.com/lbouaraba/comfyui-krea2edit) by Conrad
Locke, revision `86f886dac23013d88996e3a2e99093ba44d322fb`. It was modified to
extract only the implementation used by the Maxed Out nodes. The upstream
project is licensed under the Apache License, Version 2.0.

The complete license text is included at
`third_party/ComfyUI-Krea2Edit/LICENSE`.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

<http://www.apache.org/licenses/LICENSE-2.0>

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

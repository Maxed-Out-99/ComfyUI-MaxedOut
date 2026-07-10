"""Static checks for the web/ frontend layer. No server, no browser.

Checks:
  1. Every relative ES-module import in web/**/*.js resolves to a real file.
  2. Every node name referenced by JS (NODE_TYPE/NODE_NAME consts,
     `nodeData.name === "..."` comparisons) exists in the node schema dump.
  3. Every CSS path passed to injectCss() and every @import inside web CSS
     resolves to a real file.
  4. index.js imports every .js that lives in a subdirectory of web/
     (subdirectory files rely on index.js as their guaranteed load path).

Usage:
    python scripts/check_web.py [schema.json]
(schema defaults to scripts/node_schema_baseline.json)

Exit 0 = clean, 1 = problems found. Known-external node names (patched nodes
owned by other packs) are listed in KNOWN_EXTERNAL_NODES.
"""

import json
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(REPO_ROOT, "web")

# Node names JS may reference that are intentionally not registered by this
# repo (e.g. nodes owned by other installed packs, checked defensively).
KNOWN_EXTERNAL_NODES = set()

IMPORT_RE = re.compile(
    r"""^\s*import\s+(?:[\w${},*\s]+\s+from\s+)?["']([^"']+)["']""", re.M
)
NODE_CONST_RE = re.compile(r"""\bNODE_(?:TYPE|NAME)\s*=\s*["']([^"']+)["']""")
NODE_CMP_RE = re.compile(r"""nodeData\.name\s*===?\s*["']([^"']+)["']""")
NODE_SET_RE = re.compile(
    r"""NODE_TYPE_STRINGS[^\]]*?\[([^\]]*)\]""", re.S
)
INJECT_CSS_RE = re.compile(r"""injectCss\(\s*[`"']([^`"']+)[`"']""")
CSS_IMPORT_RE = re.compile(r"""@import\s+(?:url\()?["']?([^"')]+)["']?\)?""")


def walk(ext):
    for root, _dirs, files in os.walk(WEB):
        for f in files:
            if f.endswith(ext):
                yield os.path.join(root, f)


def rel(p):
    return os.path.relpath(p, REPO_ROOT).replace("\\", "/")


def main():
    schema_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(REPO_ROOT, "scripts", "node_schema_baseline.json")
    )
    node_keys = set()
    if os.path.exists(schema_path):
        with open(schema_path, encoding="utf-8") as f:
            node_keys = set(json.load(f)["nodes"])
    else:
        print(f"WARNING: no schema at {schema_path}; skipping node-name checks")

    problems = []
    subdir_js = set()
    index_imports = set()

    for js in walk(".js"):
        if js.endswith(".min.js"):
            continue
        with open(js, encoding="utf-8-sig") as f:
            src = f.read()

        # 1. relative imports resolve
        for target in IMPORT_RE.findall(src):
            if js.endswith("index.js") and os.path.dirname(js) == WEB:
                index_imports.add(
                    os.path.normpath(os.path.join(os.path.dirname(js), target))
                )
            if target.startswith("."):
                resolved = os.path.normpath(os.path.join(os.path.dirname(js), target))
                if resolved.startswith(WEB):
                    if not os.path.exists(resolved):
                        problems.append(f"{rel(js)}: broken import '{target}'")
                # imports escaping web/ are ComfyUI-served URLs (scripts/app.js etc.)
                elif "/scripts/" not in target.replace("\\", "/"):
                    problems.append(f"{rel(js)}: suspicious out-of-web import '{target}'")

        # 2. node names exist
        if node_keys:
            names = set(NODE_CONST_RE.findall(src)) | set(NODE_CMP_RE.findall(src))
            for m in NODE_SET_RE.findall(src):
                names.update(re.findall(r"""["']([^"']+)["']""", m))
            for name in names:
                if name not in node_keys and name not in KNOWN_EXTERNAL_NODES:
                    problems.append(f"{rel(js)}: references unknown node '{name}'")

        # 3a. injectCss paths (only checkable when repo-relative)
        for css in INJECT_CSS_RE.findall(src):
            css_rel = css.split("extensions/", 1)[-1]
            # paths are built from import.meta.url; check the tail against web/
            tail = css_rel.split("/web/", 1)[-1] if "/web/" in css_rel else None
            if tail and not os.path.exists(os.path.join(WEB, tail)):
                problems.append(f"{rel(js)}: injectCss target not found '{css}'")

        if os.path.dirname(js) != WEB:
            subdir_js.add(os.path.normpath(js))

    # 3b. CSS @imports resolve
    for css in walk(".css"):
        with open(css, encoding="utf-8-sig") as f:
            src = f.read()
        for target in CSS_IMPORT_RE.findall(src):
            if target.startswith(("http:", "https:")):
                continue
            resolved = os.path.normpath(os.path.join(os.path.dirname(css), target))
            if not os.path.exists(resolved):
                problems.append(f"{rel(css)}: broken @import '{target}'")

    # 4. subdirectory js reachable from index.js
    for js in sorted(subdir_js):
        if js not in index_imports and not js.endswith(".min.js"):
            problems.append(f"{rel(js)}: subdirectory module not imported by index.js")

    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print(f"  {p}")
        sys.exit(1)
    print("OK: web layer clean")
    sys.exit(0)


if __name__ == "__main__":
    main()

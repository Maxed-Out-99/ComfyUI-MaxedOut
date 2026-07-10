"""File/folder path helpers shared by the media and latent loaders."""
from __future__ import annotations
import os
import re
from typing import List


def _sort_paths_newest_first(paths: List[str]) -> List[str]:
    """Sort file paths by mtime desc (newest first), stable by normalized path."""
    def _mtime(path: str) -> float:
        try:
            return os.path.getmtime(path)
        except OSError:
            return 0.0

    return sorted(
        paths,
        key=lambda p: (-_mtime(p), p.replace("\\", "/").lower()),
    )


def _strip_counter(name: str) -> str:
    """Strip the trailing '_<5digits>_' counter our savers append to filenames.

    Preserves numeric-only base names like "96".
    """
    stem, _ = os.path.splitext(name)
    m = re.match(r"^(.*?)(?:_\d{5}_)$", stem)
    return m.group(1) if m else stem

"""Tolerant parsing of workflow/prompt metadata embedded in saved files."""
from __future__ import annotations
import json
from typing import Any, Dict, Optional, Union


def _safe_json_loads(s: Union[str, bytes, None]) -> Optional[Dict[str, Any]]:
    if s is None:
        return None
    if isinstance(s, bytes):
        try:
            s = s.decode("utf-8", "ignore")
        except Exception:
            return None
    if not isinstance(s, str):
        return None
    try:
        return json.loads(s)
    except Exception:
        # sometimes double-encoded in metadata
        try:
            return json.loads(json.loads(s))
        except Exception:
            return None

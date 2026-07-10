"""Guarded PromptServer route registration.

Module-scope `PromptServer.instance.routes` access crashes the whole module
import when the server isn't up (breaking node registration too, since the
root __init__ skips modules that fail to import). Registering through this
helper keeps nodes alive even if the HTTP layer is unavailable.
"""


def register_get_route(path, handler):
    """Register an async GET handler at `path`. Returns True on success."""
    try:
        from server import PromptServer
        PromptServer.instance.routes.get(path)(handler)
        return True
    except Exception as e:
        print(f"[ComfyUI-MaxedOut] Could not register route {path}: {e}")
        return False

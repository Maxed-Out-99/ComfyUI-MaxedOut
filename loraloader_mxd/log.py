import time

NAME = "ComfyUI-MaxedOut/LoraLoader-MXD"

COLORS = {
  "RESET": "\33[0m",
  "YELLOW": "\33[33m",
  "CYAN": "\33[36m",
  "RED": "\33[31m",
  "BRIGHT_GREEN": "\33[92m",
}


def _log_node(color, node_name, message):
  prefix = node_name.replace(" (MXD)", "")
  log(message, color=color, prefix=prefix)


def log_node_warn(node_name, message, msg_color="RESET"):
  _log_node("YELLOW", node_name, message)


def log_node_info(node_name, message, msg_color="RESET"):
  _log_node("CYAN", node_name, message)


def log_node_error(node_name, message, msg_color="RESET"):
  _log_node("RED", node_name, message)


def log_node_success(node_name, message, msg_color="RESET"):
  _log_node("BRIGHT_GREEN", node_name, message)


LOGGED = {}


def log(message, color=None, msg_color=None, prefix=None, id=None, at_most_secs=None):
  now = int(time.time())
  if id:
    if at_most_secs is None:
      raise ValueError("at_most_secs should be set if an id is set.")
    if id in LOGGED and now < LOGGED[id] + at_most_secs:
      return
    LOGGED[id] = now
  color_code = COLORS.get(color or "BRIGHT_GREEN", COLORS["BRIGHT_GREEN"])
  msg_color_code = COLORS.get(msg_color or "RESET", "")
  pfx = f"[{prefix}]" if prefix else ""
  print(f"{color_code}[{NAME}]{pfx}{msg_color_code} {message}{COLORS['RESET']}")

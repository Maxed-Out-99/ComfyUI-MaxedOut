import os
import shutil
from pathlib import Path

from .utils import load_json_file, path_exists, save_json_file

THIS_DIR = Path(__file__).resolve().parent
MAXEDOUT_ROOT = THIS_DIR.parent
USERDATA = MAXEDOUT_ROOT / "userdata" / "loraloader_mxd"
LEGACY_USERDATA = MAXEDOUT_ROOT.parent / "ComfyUI-LoraLoader-MXD" / "userdata"
MIGRATION_SENTINEL = USERDATA / ".migrated_from_standalone"


def _log(message: str):
  print(f"[ComfyUI-MaxedOut][LoraLoader-MXD] {message}")


def _migrate_legacy_userdata_once():
  if MIGRATION_SENTINEL.exists():
    return
  if not LEGACY_USERDATA.exists():
    return

  copied = 0
  skipped = 0
  for src in LEGACY_USERDATA.rglob("*"):
    if not src.is_file():
      continue
    rel = src.relative_to(LEGACY_USERDATA)
    dest = USERDATA / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
      skipped += 1
      continue
    shutil.copy2(src, dest)
    copied += 1

  USERDATA.mkdir(parents=True, exist_ok=True)
  MIGRATION_SENTINEL.write_text("ok\n", encoding="utf-8")
  _log(f"Migrated legacy userdata files: copied={copied}, skipped_existing={skipped}")


_migrate_legacy_userdata_once()


def read_userdata_file(rel_path: str):
  """Reads a file from the userdata directory."""
  file_path = clean_path(rel_path)
  if path_exists(file_path):
    with open(file_path, 'r', encoding='UTF-8') as file:
      return file.read()
  return None


def save_userdata_file(rel_path: str, content: str):
  """Saves a file from the userdata directory."""
  file_path = clean_path(rel_path)
  os.makedirs(os.path.dirname(file_path), exist_ok=True)
  with open(file_path, 'w+', encoding='UTF-8') as file:
    file.write(content)


def delete_userdata_file(rel_path: str):
  """Deletes a file from the userdata directory."""
  file_path = clean_path(rel_path)
  if os.path.isfile(file_path):
    os.remove(file_path)


def read_userdata_json(rel_path: str):
  """Reads a json file from the userdata directory."""
  file_path = clean_path(rel_path)
  return load_json_file(file_path)


def save_userdata_json(rel_path: str, data: dict):
  """Saves a json file from the userdata directory."""
  file_path = clean_path(rel_path)
  return save_json_file(file_path, data)


def clean_path(rel_path: str):
  """Cleans a relative path by splitting on forward slash and os.path.joining."""
  cleaned = USERDATA
  paths = rel_path.split('/')
  for path in paths:
    cleaned = cleaned / path
  return str(cleaned)

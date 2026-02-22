NAMESPACE = "mxd"


def get_name(name: str):
  return f"{name} (MXD)"


def get_category(sub_dirs=None):
  if sub_dirs is None:
    return "mxd"
  return "mxd/utils"

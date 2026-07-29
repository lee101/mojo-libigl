import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_PYTHON = os.path.join(ROOT, "python")

saved_path = sys.path[:]
sys.path = [
    path
    for path in sys.path
    if os.path.realpath(path or os.curdir) != os.path.realpath(LOCAL_PYTHON)
]
upstream_igl = importlib.import_module("igl")

for name in list(sys.modules):
    if name == "igl" or name.startswith("igl."):
        del sys.modules[name]
sys.path = saved_path
if LOCAL_PYTHON not in sys.path:
    sys.path.insert(0, LOCAL_PYTHON)
mojo_igl = importlib.import_module("igl")


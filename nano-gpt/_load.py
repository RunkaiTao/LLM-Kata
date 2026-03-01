"""
Helper to import a student's completed exercise module by section and exercise name.

Usage:
    from _load import load
    Head = load("02_layers", "b_self_attention").Head
"""
import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def load(section, exercise):
    """Load an exercise module from a given section and exercise directory."""
    path = _ROOT / section / exercise / "exercise.py"
    module_name = f"{section}.{exercise}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

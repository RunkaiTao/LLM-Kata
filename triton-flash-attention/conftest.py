import sys
import os
import importlib


def pytest_collect_file(parent, file_path):
    """Add each test file's directory to sys.path before collection."""
    test_dir = str(file_path.parent)
    if test_dir not in sys.path:
        sys.path.insert(0, test_dir)
    # Ensure 'exercise' module from this directory is importable
    # by removing any cached 'exercise' module from a different directory
    if "exercise" in sys.modules:
        existing = getattr(sys.modules["exercise"], "__file__", "")
        if existing and os.path.dirname(existing) != test_dir:
            del sys.modules["exercise"]

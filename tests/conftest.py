import os
import sys

# ----------------------------------------

from pathlib import Path

__FILE__ = os.path.abspath(__file__)

# Add src directory to Python path for imports
_project_root = Path(__file__).parent.parent
_src_path = _project_root / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))


# File: tests/conftest.py
import re

# match lines that begin with "def test_"
_pattern_test_def = re.compile(r"^def test_", re.MULTILINE)


# def pytest_collect_file(parent, file_path):
#     if not file_path.suffix == ".py":
#         return None

#     if not (
#         file_path.name.startswith("test_")
#         or file_path.name.endswith("_test.py")
#     ):
#         return None

#     content = Path(file_path).read_text()

#     if not _pattern_test_def.search(content):
#         return None

#     print(file_path)

#     return parent.session.perform_collect(file_path, parent)


def pytest_collect_file(file_path):
    # Only load files that have test functions
    if str(file_path).endswith(".py") and (
        file_path.name.startswith("test_") or file_path.name.endswith("_test.py")
    ):
        try:
            content = Path(file_path).read_text()
            if "def test_" not in content:
                return None
            print(file_path)
        except:
            pass
    return None


# def pytest_collect_file(parent, file_path):
#     # Only load files that have test functions
#     if str(file_path).endswith(".py") and (
#         file_path.name.startswith("test_")
#         or file_path.name.endswith("_test.py")
#     ):
#         try:
#             content = Path(file_path).read_text()
#             if "def test_" not in content:
#                 return None
#             print(f"Collecting tests from: {file_path}")
#             # Return the file for pytest to collect
#             return parent.session.perform_collect(file_path, parent)
#         except Exception as err:
#             print(f"Error reading {file_path}: {err}")

#     return None


# # You can also use this hook to show when a test file is actually processed
# def pytest_pycollect_makemodule(path, parent):
#     print(f"Processing module: {path}")
#     return pytest.Module.from_parent(parent, path=path)

# EOF

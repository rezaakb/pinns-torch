from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def find_root(indicator: str = ".project-root", path: Optional[os.PathLike[str]] = None) -> Path:
    """Return the nearest parent directory containing the indicator file."""
    if path is None:
        path = Path.cwd()
    else:
        path = Path(path)
    if path.is_file():
        path = path.parent

    for parent in [path, *path.parents]:
        if (parent / indicator).exists():
            return parent.resolve()
    raise FileNotFoundError(f"indicator {indicator!r} not found from {path}")


def setup_root(file: str | os.PathLike[str], indicator: str = ".project-root", pythonpath: bool = True) -> Path:
    """Set project root relative to a file and optionally add it to PYTHONPATH."""
    file_path = Path(file).resolve()
    root = find_root(indicator=indicator, path=file_path)
    if pythonpath:
        pythonpath_entries = os.environ.get("PYTHONPATH", "").split(os.pathsep)
        root_str = str(root)
        if root_str not in pythonpath_entries:
            pythonpath_entries.insert(0, root_str)
            os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, pythonpath_entries))
    return root

from __future__ import annotations

import sys
from pathlib import Path

# Centralized sys.path handling for vendor repos.
_THIRDPARTY_DIR = Path(__file__).resolve().parent


def _ensure_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def ensure_monster_paths() -> Path:
    monster_root = _THIRDPARTY_DIR / "MonSter"
    depth_anything_root = monster_root / "Depth-Anything-V2-list3"
    if not monster_root.exists():
        raise FileNotFoundError(f"MonSter repo not found at {monster_root}")
    if not depth_anything_root.exists():
        raise FileNotFoundError(f"Depth-Anything V2 not found at {depth_anything_root}")
    _ensure_sys_path(monster_root)
    _ensure_sys_path(depth_anything_root)
    return monster_root

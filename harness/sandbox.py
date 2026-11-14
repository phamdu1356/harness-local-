"""Sandbox cho tool bash: nhốt lệnh trong thư mục workspace bằng Landlock (xem _sandbox_exec.py)."""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path

_LAUNCHER = str(Path(__file__).with_name("_sandbox_exec.py"))


def available() -> bool:
    """Kernel có hỗ trợ Landlock không."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        return ctypes.CDLL(None, use_errno=True).syscall(444, None, 0, 1) >= 1
    except OSError:
        return False


def wrap(command: str, root: Path) -> list[str]:
    return [sys.executable, _LAUNCHER, str(root), "--", "bash", "-c", command]

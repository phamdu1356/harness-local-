"""Sandbox cho tool bash: nhốt lệnh trong thư mục workspace bằng Landlock (xem _sandbox_exec.py)."""
from __future__ import annotations

import ctypes
import re
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


# Tên biến môi trường trông như bí mật (token, khoá API, mật khẩu, cookie...). Chỉ là heuristic theo TÊN, không phải bảo đảm.
_SECRET_NAME = re.compile(r"token|secret|passw(or)?d|credential|api[_-]?key|private[_-]?key|access[_-]?key|auth(?!or)|cookie|_key$|^key$",
                          re.I)


def clean_env(env: dict[str, str]) -> dict[str, str]:
    """Bỏ biến môi trường trông như bí mật trước khi đưa cho lệnh do model chạy (model đọc được mọi thứ nó chạy ra).
    Muốn giữ biến nào thì liệt kê trong HARNESS_ENV_ALLOW (cách nhau bằng dấu phẩy)."""
    allow = {x.strip() for x in env.get("HARNESS_ENV_ALLOW", "").split(",") if x.strip()}
    return {k: v for k, v in env.items() if k in allow or not _SECRET_NAME.search(k)}

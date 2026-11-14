"""Các tool cho agent. Mỗi tool: schema (JSON schema kiểu OpenAI) + hàm thực thi."""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

MAX_OUTPUT = 8000


def clip(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n... [cắt {len(text) - limit} ký tự] ...\n{text[-half:]}"


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., str]
    needs_approval: bool = False

    def schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": {"type": "object", "properties": self.parameters,
                           "required": [k for k, v in self.parameters.items() if not v.get("optional")]}}}


class Workspace:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def path(self, p: str) -> Path:
        full = (self.root / p).resolve()
        if full != self.root and self.root not in full.parents:
            raise ValueError(f"Đường dẫn nằm ngoài workspace: {p}")
        return full


def build_tools(ws: Workspace, bash_timeout: int = 60) -> dict[str, Tool]:
    def read_file(path: str, offset: int = 1, limit: int = 400) -> str:
        lines = ws.path(path).read_text(errors="replace").splitlines()
        offset = max(int(offset), 1)
        chunk = lines[offset - 1: offset - 1 + int(limit)]
        out = "\n".join(f"{i}\t{l}" for i, l in enumerate(chunk, offset))
        more = len(lines) - (offset - 1 + len(chunk))
        return clip(out + (f"\n... còn {more} dòng, dùng offset" if more > 0 else ""))

    def write_file(path: str, content: str) -> str:
        p = ws.path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Đã ghi {len(content)} ký tự vào {path}"

    def edit_file(path: str, old: str, new: str) -> str:
        p = ws.path(path)
        text = p.read_text()
        n = text.count(old)
        if n == 0:
            return "LỖI: không tìm thấy `old` trong file (phải khớp chính xác, kể cả khoảng trắng)."
        if n > 1:
            return f"LỖI: `old` xuất hiện {n} lần, hãy thêm ngữ cảnh để duy nhất."
        p.write_text(text.replace(old, new, 1))
        return f"Đã sửa {path}"

    def list_dir(path: str = ".") -> str:
        p = ws.path(path)
        items = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        return clip("\n".join(f"{i.name}{'/' if i.is_dir() else ''}" for i in items if i.name != ".git")) or "(trống)"

    def grep(pattern: str, path: str = ".") -> str:
        rx, hits = re.compile(pattern), []
        base = ws.path(path)
        files = [base] if base.is_file() else (
            f for f in base.rglob("*") if f.is_file()
            and not any(part in {".git", "node_modules", "__pycache__", ".venv"} for part in f.parts))
        for f in files:
            try:
                for i, line in enumerate(f.read_text().splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{f.relative_to(ws.root)}:{i}: {line.strip()[:200]}")
                        if len(hits) >= 100:
                            return clip("\n".join(hits)) + "\n(đủ 100 kết quả)"
            except (UnicodeDecodeError, OSError):
                continue
        return clip("\n".join(hits)) or "Không có kết quả."

    def bash(command: str) -> str:
        try:
            r = subprocess.run(command, shell=True, cwd=ws.root, capture_output=True,
                               text=True, timeout=bash_timeout, env={**os.environ, "PAGER": "cat"})
        except subprocess.TimeoutExpired:
            return f"LỖI: quá {bash_timeout}s, đã dừng."
        out = (r.stdout + (f"\n[stderr]\n{r.stderr}" if r.stderr else "")).strip()
        return clip(f"{out}\n[exit {r.returncode}]")

    S = {"type": "string"}
    I = {"type": "integer", "optional": True}
    tools = [
        Tool("read_file", "Đọc file, có số dòng.", {"path": S, "offset": I, "limit": I}, read_file),
        Tool("list_dir", "Liệt kê thư mục.", {"path": {**S, "optional": True}}, list_dir),
        Tool("grep", "Tìm regex trong file/thư mục.", {"pattern": S, "path": {**S, "optional": True}}, grep),
        Tool("write_file", "Ghi đè/tạo file.", {"path": S, "content": S}, write_file, True),
        Tool("edit_file", "Thay đúng một đoạn `old` bằng `new` trong file.",
             {"path": S, "old": S, "new": S}, edit_file, True),
        Tool("bash", "Chạy lệnh shell trong workspace.", {"command": S}, bash, True),
    ]
    return {t.name: t for t in tools}

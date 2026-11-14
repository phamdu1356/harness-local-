"""Các tool cho agent: schema (kiểu OpenAI) + hàm thực thi + kiểm tra tham số."""
from __future__ import annotations

import fnmatch
import html
import os
import re
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

MAX_OUTPUT = 8000
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".mypy_cache"}


def clip(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n... [cắt {len(text) - limit} ký tự] ...\n{text[-half:]}"


def P(type_: str, desc: str, optional: bool = False) -> dict:
    return {"type": type_, "description": desc, "optional": optional}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., str]
    needs_approval: bool = False
    read_only: bool = False

    def schema(self) -> dict:
        props = {k: {kk: vv for kk, vv in v.items() if kk != "optional"} for k, v in self.parameters.items()}
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": {"type": "object", "properties": props,
                           "required": [k for k, v in self.parameters.items() if not v.get("optional")]}}}

    def clean_args(self, args: dict) -> dict | str:
        """Trả về args đã chuẩn hoá, hoặc chuỗi LỖI để model tự sửa."""
        if not isinstance(args, dict):
            return f"LỖI: arguments phải là object JSON, nhận {type(args).__name__}."
        missing = [k for k, v in self.parameters.items() if not v.get("optional") and k not in args]
        if missing:
            sig = ", ".join(f"{k}{'?' if v.get('optional') else ''}: {v['type']}" for k, v in self.parameters.items())
            return f"LỖI: thiếu tham số {', '.join(missing)}. Chữ ký: {self.name}({sig})"
        out = {}
        for k, v in args.items():
            spec = self.parameters.get(k)
            if not spec:
                continue  # bỏ tham số lạ
            try:
                out[k] = int(v) if spec["type"] == "integer" else (v if spec["type"] != "string" else str(v))
            except (TypeError, ValueError):
                return f"LỖI: tham số {k} phải là {spec['type']}, nhận {v!r}."
        return out


class Workspace:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def path(self, p: str) -> Path:
        full = (self.root / p).resolve()
        if full != self.root and self.root not in full.parents:
            raise ValueError(f"Đường dẫn nằm ngoài workspace: {p}")
        return full

    def rel(self, p: Path) -> str:
        return str(p.relative_to(self.root))


def _walk(base: Path):
    for dirpath, dirs, files in os.walk(base):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for f in sorted(files):
            yield Path(dirpath) / f


def build_tools(ws: Workspace, bash_timeout: int = 60) -> dict[str, Tool]:
    todo_state: list[str] = []

    def read_file(path: str, offset: int = 1, limit: int = 400) -> str:
        p = ws.path(path)
        if p.stat().st_size > 2_000_000:
            return "LỖI: file quá lớn (>2MB). Dùng grep hoặc bash (head/sed)."
        lines = p.read_text(errors="replace").splitlines()
        offset = max(offset, 1)
        chunk = lines[offset - 1: offset - 1 + limit]
        out = "\n".join(f"{i}\t{l}" for i, l in enumerate(chunk, offset))
        more = len(lines) - (offset - 1 + len(chunk))
        return clip(out + (f"\n... còn {more} dòng, dùng offset" if more > 0 else "")) or "(file trống)"

    def write_file(path: str, content: str) -> str:
        p = ws.path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        existed = p.exists()
        p.write_text(content)
        return f"Đã {'ghi đè' if existed else 'tạo'} {path} ({len(content.splitlines())} dòng)"

    def edit_file(path: str, old: str, new: str) -> str:
        p = ws.path(path)
        text = p.read_text()
        n = text.count(old)
        if n == 0:
            return "LỖI: không tìm thấy `old` (phải khớp chính xác, kể cả khoảng trắng). Đọc lại file bằng read_file."
        if n > 1:
            return f"LỖI: `old` xuất hiện {n} lần, thêm ngữ cảnh để duy nhất."
        p.write_text(text.replace(old, new, 1))
        return f"Đã sửa {path}"

    def list_dir(path: str = ".") -> str:
        items = sorted(ws.path(path).iterdir(), key=lambda x: (not x.is_dir(), x.name))
        return clip("\n".join(f"{i.name}{'/' if i.is_dir() else ''}" for i in items if i.name != ".git")) or "(trống)"

    def glob(pattern: str, path: str = ".") -> str:
        base = ws.path(path)
        pat = pattern if "/" in pattern or pattern.startswith("**") else f"**/{pattern}"
        hits = [ws.rel(f) for f in _walk(base) if fnmatch.fnmatch(ws.rel(f), pat) or fnmatch.fnmatch(f.name, pattern)]
        return clip("\n".join(hits[:200])) + (f"\n... {len(hits)} kết quả" if len(hits) > 200 else "") if hits else "Không có file khớp."

    def grep(pattern: str, path: str = ".", include: str = "") -> str:
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"LỖI regex: {e}"
        base, hits = ws.path(path), []
        for f in ([base] if base.is_file() else _walk(base)):
            if include and not fnmatch.fnmatch(f.name, include):
                continue
            try:
                if f.stat().st_size > 1_000_000:
                    continue
                for i, line in enumerate(f.read_text().splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{ws.rel(f)}:{i}: {line.strip()[:200]}")
                        if len(hits) >= 100:
                            return clip("\n".join(hits)) + "\n(đủ 100 kết quả)"
            except (UnicodeDecodeError, OSError):
                continue
        return clip("\n".join(hits)) or "Không có kết quả."

    def move_file(src: str, dst: str) -> str:
        d = ws.path(dst)
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(ws.path(src)), str(d))
        return f"Đã chuyển {src} → {dst}"

    def delete_file(path: str) -> str:
        p = ws.path(path)
        if p == ws.root:
            return "LỖI: không xoá workspace."
        shutil.rmtree(p) if p.is_dir() else p.unlink()
        return f"Đã xoá {path}"

    def bash(command: str) -> str:
        try:
            r = subprocess.run(command, shell=True, cwd=ws.root, capture_output=True, text=True,
                               timeout=bash_timeout, stdin=subprocess.DEVNULL, env={**os.environ, "PAGER": "cat"})
        except subprocess.TimeoutExpired:
            return f"LỖI: quá {bash_timeout}s, đã dừng."
        out = (r.stdout + (f"\n[stderr]\n{r.stderr}" if r.stderr else "")).strip()
        return clip(f"{out}\n[exit {r.returncode}]")

    def web_fetch(url: str) -> str:
        if not re.match(r"https?://", url):
            return "LỖI: chỉ hỗ trợ http(s)."
        req = urllib.request.Request(url, headers={"User-Agent": "harness/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                raw = r.read(1_500_000).decode(r.headers.get_content_charset() or "utf-8", errors="replace")
                ctype = r.headers.get("Content-Type", "")
        except Exception as e:
            return f"LỖI tải trang: {e}"
        if "html" in ctype:
            raw = re.sub(r"(?is)<(script|style|noscript|svg).*?</\1>", " ", raw)
            raw = re.sub(r"(?s)<[^>]+>", " ", raw)
            raw = re.sub(r"[ \t]+", " ", html.unescape(raw))
            raw = re.sub(r"\n\s*\n+", "\n\n", raw)
        return clip(raw.strip(), 6000)

    def todo(todos: str) -> str:
        """Danh sách việc: mỗi dòng '[ ] việc' / '[~] đang làm' / '[x] xong'."""
        todo_state[:] = [l.rstrip() for l in todos.splitlines() if l.strip()]
        done = sum(1 for l in todo_state if l.lstrip().startswith("[x]"))
        return f"Đã cập nhật kế hoạch ({done}/{len(todo_state)} xong):\n" + "\n".join(todo_state)

    S, I = "string", "integer"
    tools = [
        Tool("read_file", "Đọc file text, có số dòng. File dài: dùng offset/limit.",
             {"path": P(S, "đường dẫn tương đối"), "offset": P(I, "dòng bắt đầu (từ 1)", True),
              "limit": P(I, "số dòng tối đa, mặc định 400", True)}, read_file, read_only=True),
        Tool("list_dir", "Liệt kê một thư mục.", {"path": P(S, "mặc định '.'", True)}, list_dir, read_only=True),
        Tool("glob", "Tìm file theo tên/mẫu, vd '*.py' hoặc 'src/**/*.ts'.",
             {"pattern": P(S, "mẫu glob"), "path": P(S, "thư mục gốc", True)}, glob, read_only=True),
        Tool("grep", "Tìm regex trong nội dung file, trả về file:dòng.",
             {"pattern": P(S, "regex Python"), "path": P(S, "file/thư mục", True),
              "include": P(S, "lọc tên file, vd '*.py'", True)}, grep, read_only=True),
        Tool("todo", "Ghi/cập nhật kế hoạch cho việc nhiều bước. Mỗi dòng: '[ ] việc', '[~] đang làm', '[x] xong'.",
             {"todos": P(S, "toàn bộ danh sách, mỗi việc một dòng")}, todo, read_only=True),
        Tool("write_file", "Tạo file mới hoặc ghi đè toàn bộ file.",
             {"path": P(S, "đường dẫn"), "content": P(S, "nội dung đầy đủ")}, write_file, True),
        Tool("edit_file", "Thay đúng một đoạn `old` (duy nhất) bằng `new`.",
             {"path": P(S, "đường dẫn"), "old": P(S, "đoạn gốc, khớp chính xác"), "new": P(S, "đoạn thay thế")},
             edit_file, True),
        Tool("move_file", "Đổi tên/di chuyển file hoặc thư mục.",
             {"src": P(S, "nguồn"), "dst": P(S, "đích")}, move_file, True),
        Tool("delete_file", "Xoá file hoặc thư mục.", {"path": P(S, "đường dẫn")}, delete_file, True),
        Tool("bash", "Chạy lệnh shell trong workspace (không tương tác, timeout 60s).",
             {"command": P(S, "lệnh shell")}, bash, True),
        Tool("web_fetch", "Tải một trang web và trả về văn bản thuần.",
             {"url": P(S, "URL http(s)")}, web_fetch, True),
    ]
    return {t.name: t for t in tools}

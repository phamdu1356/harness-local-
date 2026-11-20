"""Các tool cho agent: schema (kiểu OpenAI) + hàm thực thi + kiểm tra tham số."""
from __future__ import annotations

import difflib
import fnmatch
import html
import json
import os
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import sandbox as _sandbox

MAX_OUTPUT = 8000
SKIP_DIRS = {".git", ".harness", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".mypy_cache"}


def clip(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n... [cắt {len(text) - limit} ký tự] ...\n{text[-half:]}"


def closest_block(text: str, old: str, context: int = 1) -> str:
    """Đoạn trong text giống `old` nhất (theo dòng), kèm số dòng; rỗng nếu không có gì đủ giống."""
    lines, want = text.splitlines(), old.splitlines() or [old]
    n = len(want)
    if not lines:
        return ""
    best, best_i = 0.0, 0
    for i in range(max(1, len(lines) - n + 1)):
        r = difflib.SequenceMatcher(None, "\n".join(lines[i:i + n]), "\n".join(want)).ratio()
        if r > best:
            best, best_i = r, i
    if best < 0.6:
        return ""
    lo, hi = max(0, best_i - context), min(len(lines), best_i + n + context)
    return "\n".join(f"{j + 1}\t{lines[j]}" for j in range(lo, hi))


def P(type_: str, desc: str, optional: bool = False, nullable: bool = False) -> dict:
    d = {"type": type_, "description": desc, "optional": optional}
    if nullable:
        d["nullable"] = True
    return d


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., str]
    needs_approval: bool = False
    read_only: bool = False

    def schema(self) -> dict:
        props = {k: {kk: vv for kk, vv in v.items() if kk not in ("optional", "nullable")} for k, v in self.parameters.items()}
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
            if v is None:
                if spec.get("nullable"):
                    out[k] = None
                    continue
                if spec.get("optional"):
                    continue
                return f"LỖI: tham số {k} phải là {spec['type']}, nhận None."
            if spec["type"] == "integer" and isinstance(v, bool):
                return f"LỖI: tham số {k} phải là {spec['type']}, nhận {v!r}."
            try:
                out[k] = int(v) if spec["type"] == "integer" else (v if spec["type"] != "string" else str(v))
            except (TypeError, ValueError):
                return f"LỖI: tham số {k} phải là {spec['type']}, nhận {v!r}."
        return out


def parse_search_results(page: str, num: int = 5) -> str:
    """Trích tiêu đề, URL, đoạn trích từ trang kết quả html.duckduckgo.com."""
    def text(x: str) -> str:
        return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", x))).strip()

    out = []
    blocks = re.split(r'<div[^>]+class="[^"]*\bresult\b[^"]*"', page)[1:]
    for b in blocks:
        a = re.search(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', b, re.S)
        if not a:
            continue
        href = html.unescape(a.group(1))
        m = re.search(r"uddg=([^&]+)", href)
        url = urllib.parse.unquote(m.group(1)) if m else href
        sn = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', b, re.S)
        out.append(f"{len(out) + 1}. {text(a.group(2))}\n   {url}" + (f"\n   {clip(text(sn.group(1)), 240)}" if sn else ""))
        if len(out) >= num:
            break
    return "\n".join(out)


def format_searx(data: dict, num: int = 5) -> str:
    out = []
    for r in (data.get("results") or [])[:num]:
        out.append(f"{len(out) + 1}. {r.get('title', '').strip()}\n   {r.get('url', '')}"
                   + (f"\n   {clip(r['content'].strip(), 240)}" if r.get("content") else ""))
    return "\n".join(out)


def _allowed_path(p: Path, root: Path) -> bool:
    try:
        resolved = p.resolve()
    except (OSError, RuntimeError):
        return False
    if resolved != root and not resolved.is_relative_to(root):
        return False
    rel = resolved.relative_to(root)
    return not (rel.parts and rel.parts[0] == ".harness")


class Workspace:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def path(self, p: str) -> Path:
        try:
            full = (self.root / p).resolve()
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Đường dẫn không hợp lệ ({p}): {e}") from e
        if full != self.root and not full.is_relative_to(self.root):
            raise ValueError(f"Đường dẫn nằm ngoài workspace: {p}")
        rel = full.relative_to(self.root)
        if rel.parts and rel.parts[0] == ".harness":
            raise ValueError(f"Không được truy cập thư mục nội bộ .harness/: {p}")
        return full

    def rel(self, p: Path) -> str:
        return str(p.relative_to(self.root))


def _walk(base: Path, root: Path | None = None):
    root = (root or base).resolve()
    for dirpath, dirs, files in os.walk(base):
        dirs[:] = [d for d in sorted(dirs) if d not in SKIP_DIRS and _allowed_path(Path(dirpath) / d, root)]
        for f in sorted(files):
            fp = Path(dirpath) / f
            if _allowed_path(fp, root):
                yield fp


def build_tools(ws: Workspace, bash_timeout: int = 60, sandbox: bool = True) -> dict[str, Tool]:
    todo_state: list[str] = []
    seen_state: dict[str, tuple[int, int]] = {}  # file -> (mtime_ns, size) tại lần model đọc/ghi gần nhất

    def mark(p: Path) -> None:
        st = p.stat()
        seen_state[str(p)] = (st.st_mtime_ns, st.st_size)

    def read_file(path: str, offset: int = 1, limit: int = 400) -> str:
        p = ws.path(path)
        if p.stat().st_size > 2_000_000:
            return "LỖI: file quá lớn (>2MB). Dùng grep hoặc bash (head/sed)."
        mark(p)
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
        mark(p)
        return f"Đã {'ghi đè' if existed else 'tạo'} {path} ({len(content.splitlines())} dòng)"

    def edit_file(path: str, old: str, new: str) -> str:
        p = ws.path(path)
        text = p.read_text()
        n = text.count(old)
        if n == 0:
            hint = closest_block(text, old)
            st = p.stat()
            stale = str(p) in seen_state and seen_state[str(p)] != (st.st_mtime_ns, st.st_size)
            return ("LỖI: không tìm thấy `old` (phải khớp chính xác, kể cả khoảng trắng và thụt lề)."
                    + ("\nLƯU Ý: file này ĐÃ THAY ĐỔI kể từ lần bạn đọc/ghi gần nhất (có thể do lệnh bash/sed của bạn), "
                       "nên bản bạn đang nhớ đã cũ. Hãy read_file lại rồi mới sửa." if stale else "")
                    + (f"\nĐoạn gần giống nhất trong file (số dòng\tnội dung):\n{hint}\nChép đúng nội dung thật ở trên vào `old`."
                       if hint else " Đọc lại file bằng read_file."))
        if n > 1:
            return f"LỖI: `old` xuất hiện {n} lần, thêm ngữ cảnh để duy nhất."
        p.write_text(text.replace(old, new, 1))
        mark(p)
        return f"Đã sửa {path}"

    def list_dir(path: str = ".") -> str:
        base = ws.path(path)
        items = [i for i in base.iterdir() if i.name not in (".git", ".harness") and _allowed_path(i, ws.root)]
        items.sort(key=lambda x: (not x.is_dir(), x.name))
        return clip("\n".join(f"{i.name}{'/' if i.is_dir() else ''}" for i in items)) or "(trống)"

    def glob(pattern: str, path: str = ".") -> str:
        base = ws.path(path)
        pat = pattern if "/" in pattern or pattern.startswith("**") else f"**/{pattern}"
        hits = [ws.rel(f) for f in _walk(base, ws.root) if fnmatch.fnmatch(ws.rel(f), pat) or fnmatch.fnmatch(f.name, pattern)]
        return clip("\n".join(hits[:200])) + (f"\n... {len(hits)} kết quả" if len(hits) > 200 else "") if hits else "Không có file khớp."

    def grep(pattern: str, path: str = ".", include: str = "") -> str:
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"LỖI regex: {e}"
        base, hits = ws.path(path), []
        for f in ([base] if base.is_file() else _walk(base, ws.root)):
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
        s, d = ws.path(src), ws.path(dst)
        if s == ws.root:
            return "LỖI: không di chuyển workspace."
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(s), str(d))
        return f"Đã chuyển {src} → {dst}"

    def delete_file(path: str) -> str:
        p = ws.path(path)
        if p == ws.root:
            return "LỖI: không xoá workspace."
        shutil.rmtree(p) if p.is_dir() else p.unlink()
        return f"Đã xoá {path}"

    def bash(command: str) -> str:
        env = {**os.environ, "PAGER": "cat"}
        if sandbox:
            env = _sandbox.clean_env(env)   # model đọc được mọi thứ lệnh của nó in ra: đừng đưa token/khoá API của bạn vào
            if not _sandbox.available():  # đóng kín: không sandbox được thì không chạy
                return ("LỖI: không dựng được sandbox (kernel thiếu Landlock) nên bash bị tắt để giữ giới hạn "
                        "thư mục. Chạy với --no-sandbox nếu chấp nhận rủi ro.")
            argv, shell = _sandbox.wrap(command, ws.root), False
        else:
            argv, shell = command, True
        try:
            r = subprocess.run(argv, shell=shell, cwd=ws.root, capture_output=True, text=True,
                               timeout=bash_timeout, stdin=subprocess.DEVNULL, env=env)
        except subprocess.TimeoutExpired:
            return f"LỖI: quá {bash_timeout}s, đã dừng."
        out = (r.stdout + (f"\n[stderr]\n{r.stderr}" if r.stderr else "")).strip()
        if sandbox and re.search(r"Permission denied|Không đủ quyền|Operation not permitted", out):
            out += ("\n[sandbox] Bạn chỉ truy cập được bên trong thư mục workspace (đây cũng là thư mục hiện tại của "
                    "bash). Dùng đường dẫn TƯƠNG ĐỐI như `src/a.py` hoặc `.`, đừng dùng đường dẫn tuyệt đối hay cd ra ngoài.")
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

    def web_search(query: str, num: int = 5) -> str:
        num = max(1, min(num, 10))
        if searx := os.environ.get("HARNESS_SEARX_URL"):  # SearXNG của bạn (API JSON), ổn định hơn
            url = searx.rstrip("/") + "/search?" + urllib.parse.urlencode({"q": query, "format": "json"})
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "harness/0.1"}),
                                            timeout=20) as r:
                    return format_searx(json.load(r), num) or "Không có kết quả."
            except Exception as e:
                return f"LỖI tìm kiếm (SearXNG): {e}"
        url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "harness/0.1"}),
                                        timeout=15) as r:
                status, page = r.status, r.read(1_000_000).decode("utf-8", errors="replace")
        except Exception as e:
            return f"LỖI tìm kiếm: {e}"
        out = parse_search_results(page, num)
        if not out and (status == 202 or re.search(r"anomaly|captcha", page, re.I)):
            return ("LỖI: DuckDuckGo đang chặn truy vấn tự động. Thử lại sau, hoặc đặt HARNESS_SEARX_URL trỏ tới "
                    "một SearXNG của bạn, hoặc dùng web_fetch với một URL cụ thể.")
        return out or "Không có kết quả."

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
        Tool("web_search", "Tìm kiếm web, trả về tiêu đề, URL và đoạn trích. Dùng web_fetch để đọc kỹ một kết quả.",
             {"query": P(S, "từ khoá tìm kiếm"), "num": P(I, "số kết quả, mặc định 5", True)}, web_search,
             needs_approval=True, read_only=True),   # câu truy vấn rời máy bạn: --ask phải hỏi
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
        Tool("bash", "Chạy lệnh shell trong workspace (không tương tác, timeout 60s). Bị nhốt trong thư mục workspace: không đọc/ghi được gì ngoài đó.",
             {"command": P(S, "lệnh shell")}, bash, True),
        Tool("web_fetch", "Tải một trang web và trả về văn bản thuần.",
             {"url": P(S, "URL http(s)")}, web_fetch, True),
    ]
    for t in tools:
        if t.name == "todo":
            t.state = todo_state   # để chế độ --goal đọc/ghi kế hoạch
    return {t.name: t for t in tools}

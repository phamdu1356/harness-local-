"""Giao diện terminal: màu ANSI, khung, spinner. Tự tắt màu khi không phải TTY hoặc có NO_COLOR."""
from __future__ import annotations

import itertools
import json
import os
import shutil
import sys
import threading
import time

from .tools import clip

RESET = "\033[0m"
STYLES = {
    "dim": "\033[2m", "bold": "\033[1m", "italic": "\033[3m",
    "accent": "\033[38;5;75m", "user": "\033[38;5;114m", "tool": "\033[38;5;179m",
    "ok": "\033[38;5;114m", "err": "\033[38;5;203m", "warn": "\033[38;5;215m",
    "think": "\033[2;3;38;5;245m", "muted": "\033[38;5;245m",
}


class UI:
    def __init__(self, out=None, color: bool | None = None):
        self.out = out or sys.stdout
        tty = hasattr(self.out, "isatty") and self.out.isatty()
        self.color = tty and "NO_COLOR" not in os.environ if color is None else color
        self.tty = tty
        self._spin: threading.Event | None = None
        self._phase = None  # None | "think" | "answer"

    # --- cơ bản ---
    def c(self, style: str, text: str) -> str:
        return f"{STYLES[style]}{text}{RESET}" if self.color else text

    def w(self, s: str = "", end: str = "\n"):
        print(s, end=end, file=self.out, flush=True)

    def width(self) -> int:
        return min(shutil.get_terminal_size((90, 20)).columns, 100)

    # --- spinner ---
    def spinner_start(self, label: str = "đang nghĩ"):
        if not self.tty or self._spin:
            return
        stop = self._spin = threading.Event()

        def run():
            t0 = time.time()
            for f in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
                if stop.is_set():
                    break
                self.out.write(f"\r{self.c('accent', f)} {self.c('muted', f'{label} {time.time() - t0:.0f}s')}")
                self.out.flush()
                time.sleep(0.08)
            self.out.write("\r\033[2K")
            self.out.flush()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def spinner_stop(self):
        if self._spin:
            self._spin.set()
            self._thread.join()
            self._spin = None

    # --- thành phần ---
    def banner(self, model: str, cwd: str):
        line = "─" * min(self.width(), 60)
        self.w(self.c("accent", f"╭{line}"))
        self.w(f"{self.c('accent', '│')} {self.c('bold', 'harness')} {self.c('muted', 'agent cho LLM local')}")
        self.w(f"{self.c('accent', '│')} {self.c('muted', 'model')} {model}")
        self.w(f"{self.c('accent', '│')} {self.c('muted', 'thư mục')} {cwd}")
        self.w(f"{self.c('accent', '│')} {self.c('dim', '/clear · /context · /memory · /forget · /exit')}")
        self.w(self.c("accent", f"╰{line}"))

    def prompt(self) -> str:
        return f"\n{self.c('user', '❯')} "

    def stream_think(self, text: str):
        if self._phase != "think":
            self.spinner_stop()
            self.w(self.c("muted", "┊ suy nghĩ"))
            self._phase = "think"
        self.w(self.c("think", text.replace("\n", "\n")), end="")

    def stream_answer(self, text: str):
        if self._phase != "answer":
            self.spinner_stop()
            if self._phase == "think":
                self.w("\n")
            self.w(self.c("accent", "◆ "), end="")
            self._phase = "answer"
            text = text.lstrip("\n")
        self.w(text, end="")

    def end_stream(self):
        self.spinner_stop()
        if self._phase:
            self.w()
        self._phase = None

    def tool_call(self, name: str, args: dict):
        self.end_stream()
        main = next(iter(args.values()), "") if args else ""
        summary = clip(str(main).replace("\n", " ⏎ "), 90) if len(args) <= 1 else clip(
            json.dumps(args, ensure_ascii=False).replace("\\n", " ⏎ "), 90)
        self.w(f"{self.c('tool', '▸')} {self.c('bold', name)} {self.c('muted', summary)}")

    def tool_result(self, result: str, max_lines: int = 6):
        failed = result.startswith("LỖI") or "Người dùng từ chối" in result
        lines = result.strip().splitlines() or ["(trống)"]
        shown = lines[:max_lines]
        mark = self.c("err" if failed else "ok", "✗" if failed else "✓")
        for i, l in enumerate(shown):
            gutter = f" {mark} " if i == 0 else "   "
            self.w(gutter + self.c("err" if failed else "dim", clip(l, self.width() - 6)))
        if len(lines) > max_lines:
            self.w("   " + self.c("muted", f"… +{len(lines) - max_lines} dòng"))

    def ask_approval(self, name: str, args: dict) -> str:
        detail = args.get("command") or args.get("path") or json.dumps(args, ensure_ascii=False)
        self.end_stream()
        return f"{self.c('warn', '⚠')} cho phép {self.c('bold', name)} {self.c('muted', clip(str(detail), 120))} {self.c('dim', '[y/N]')} "

    def memory_note(self, n: int):
        self.end_stream()
        self.w(self.c("muted", f"◌ nhớ lại {n} mục từ bộ nhớ ngoài"))

    def summary_note(self):
        self.end_stream()
        self.w(self.c("muted", "◌ đã tóm tắt hội thoại cũ để giải phóng context"))

    def verify_note(self, ok: bool, k: int, n: int):
        self.end_stream()
        self.w(self.c("ok", "✓ kiểm tra tự động: đạt") if ok else self.c("err", f"✗ kiểm tra tự động thất bại (vòng {k}/{n}): agent phải sửa tiếp"))

    def footer(self, seconds: float, steps: int, ctx: str = ""):
        self.w(self.c("dim", f"  ⏱ {seconds:.1f}s · {steps} lượt" + (f" · ctx {ctx}" if ctx else "")))

    def error(self, msg: str):
        self.w(self.c("err", f"✗ {msg}"))

from __future__ import annotations

import argparse
import os
import sys

from .agent import Agent
from .client import LLMError, LMStudioClient
from .ui import UI
from .tools import Workspace, build_tools


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="harness", description="Agent coding CLI cho LLM local (LM Studio)")
    ap.add_argument("-m", "--model", default=os.environ.get("HARNESS_MODEL"),
                    help="id model trong LM Studio (mặc định: model đang load)")
    ap.add_argument("--host", default=os.environ.get("LMSTUDIO_HOST", "http://localhost:1234"))
    ap.add_argument("-C", "--cwd", default=".", help="workspace (mặc định: thư mục hiện tại)")
    ap.add_argument("-p", "--prompt", help="chạy một lần rồi thoát")
    ap.add_argument("-y", "--yes", action="store_true", help="tự duyệt mọi tool (write/edit/bash)")
    ap.add_argument("--text-tools", action="store_true",
                    help="model không hỗ trợ tool-calling native: mô tả tool trong prompt")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-steps", type=int, default=25)
    a = ap.parse_args(argv)
    if not a.host.startswith("http"):
        a.host = "http://" + a.host

    ws = Workspace(a.cwd)
    client = LMStudioClient(a.model, a.host, a.temperature)
    ui = UI()
    agent = Agent(client, build_tools(ws), ws, ui=ui, auto_approve=a.yes,
                  text_tools=a.text_tools, max_steps=a.max_steps)
    try:
        model = client.ensure_model()
        if a.prompt:
            agent.run(a.prompt)
            return 0
        ui.banner(model, str(ws.root))
        while True:
            try:
                line = input(ui.prompt()).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not line:
                continue
            if line in ("/exit", "/quit"):
                return 0
            if line == "/clear":
                agent.reset()
                ui.w(ui.c("muted", "Đã xoá hội thoại."))
                continue
            try:
                agent.run(line)
            except KeyboardInterrupt:
                ui.w(ui.c("muted", "\n[đã ngắt]"))
    except LLMError as e:
        ui.error(str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())

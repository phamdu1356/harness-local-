from __future__ import annotations

import argparse
import os
import sys
import uuid

from .agent import Agent
from .client import LLMError, LMStudioClient
from .embed import EmbedClient
from .memory import Memory, memory_tools
from .serve import EmbedServer, LocalServer, ServeError, find_embed_model, find_model
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
    ap.add_argument("--local", nargs="?", const="", metavar="MODEL",
                    help="tự chạy llama-server (không cần LM Studio). MODEL: đường dẫn .gguf hoặc một phần tên")
    ap.add_argument("--cpu", action="store_true", help="chạy model hoàn toàn bằng CPU (ngụ ý --local, không dùng GPU)")
    ap.add_argument("--threads", type=int, default=0, help="số luồng CPU khi dùng --cpu (mặc định: 8)")
    ap.add_argument("--ctx", type=int, default=16384, help="context (dùng --local, và làm ngân sách dọn context)")
    ap.add_argument("--port", type=int, default=8081, help="cổng khi dùng --local")
    ap.add_argument("--think", choices=["on", "off", "auto"], default="auto",
                    help="bật/tắt chế độ suy nghĩ khi dùng --local")
    ap.add_argument("--embed", metavar="MODEL", help="model embedding .gguf (đường dẫn hoặc một phần tên); mặc định tự tìm")
    ap.add_argument("--no-embed", action="store_true", help="không dùng embedding (chỉ tìm theo từ khoá BM25)")
    ap.add_argument("--no-memory", action="store_true", help="tắt bộ nhớ ngoài")
    ap.add_argument("--system", help="prompt bổ sung (văn bản, hoặc @đường/dẫn/file)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-steps", type=int, default=25)
    a = ap.parse_args(argv)
    if not a.host.startswith("http"):
        a.host = "http://" + a.host

    server = None
    if a.cpu and a.local is None:
        a.local = ""
    if a.local is not None:
        try:
            server = LocalServer(find_model(a.local or None), a.port, a.ctx, reasoning=a.think,
                                 cpu=a.cpu, threads=(a.threads or 8) if a.cpu else 0)
            print(f"Khởi động llama-server ({'CPU' if a.cpu else 'GPU'}) · {server.model.name} · ctx {a.ctx} ... (nạp model có thể mất 10–60s)", flush=True)
            server.start()
        except ServeError as e:
            print(f"✗ {e}", file=sys.stderr)
            return 1
        a.host, a.model = server.host, a.model or "local"
    embed_server = None
    a.embedder = None
    if not a.no_memory and not a.no_embed:
        try:
            embed_server = EmbedServer(find_embed_model(a.embed), a.port + 1)
            print(f"Khởi động embedding (CPU) · {embed_server.model.name} ...", flush=True)
            embed_server.start()
            a.embedder = EmbedClient(embed_server.host)
        except ServeError as e:
            print(f"⚠ không bật embedding ({str(e).splitlines()[0]}); dùng tìm kiếm từ khoá.", file=sys.stderr)
            embed_server = None
    try:
        return run(a)
    finally:
        for srv in (server, embed_server):
            if srv:
                srv.stop()


def run(a) -> int:
    ws = Workspace(a.cwd)
    client = LMStudioClient(a.model, a.host, a.temperature)
    ui = UI()
    memory = None if a.no_memory else Memory(ws.root / ".harness" / "memory.db", uuid.uuid4().hex[:8], a.embedder)
    extra = a.system or ""
    if extra.startswith("@"):
        extra = open(extra[1:]).read()
    if memory and memory.embedder:
        n = memory.backfill()
        if n:
            ui.w(ui.c("muted", f"◌ đã nhúng bù {n} mục cũ vào bộ nhớ"))
    tools = build_tools(ws)
    agent = Agent(client, tools, ws, ui=ui, memory=memory, max_context_chars=int(a.ctx * 2.2), auto_approve=a.yes,
                  text_tools=a.text_tools, system_extra=extra, max_steps=a.max_steps)
    if memory:
        tools.update(memory_tools(memory, lambda: agent.turn))
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
            if line == "/memory":
                st = memory.stats() if memory else None
                ui.w(ui.c("muted", "Bộ nhớ ngoài: " + (", ".join(f"{k} {v}" for k, v in st.items()) or "trống") if memory else "Bộ nhớ ngoài đang tắt."))
                continue
            if line == "/forget":
                if memory:
                    memory.clear()
                ui.w(ui.c("muted", "Đã xoá bộ nhớ ngoài."))
                continue
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

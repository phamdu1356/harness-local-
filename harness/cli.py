from __future__ import annotations

import argparse
import contextlib
import os
import re
import sys
import uuid

from .agent import Agent, delegate_tool
from .client import LLMError, LMStudioClient
from .embed import EmbedClient
from .mcp import load_mcp_tools
from .memory import Memory, memory_tools
from .serve import EmbedServer, LocalServer, ServeError, find_embed_model, find_model
from .tools import Workspace, build_tools
from .ui import UI

DEFAULT_CTX = 16384
CTX_USABLE = 0.5   # phần context dành cho hội thoại; context nhỏ => mỗi bước nhanh hơn rõ rệt, phần cũ đã có bộ nhớ ngoài


def common_parser() -> argparse.ArgumentParser:
    """Các cờ dùng chung cho REPL và eval."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("-m", "--model", default=os.environ.get("HARNESS_MODEL"),
                    help="id model trong LM Studio (mặc định: model đang load)")
    ap.add_argument("--host", default=os.environ.get("LMSTUDIO_HOST", "http://localhost:1234"))
    ap.add_argument("--text-tools", action="store_true",
                    help="model không hỗ trợ tool-calling native: mô tả tool trong prompt")
    ap.add_argument("--local", nargs="?", const="", metavar="MODEL",
                    help="tự chạy llama-server (không cần LM Studio). MODEL: đường dẫn .gguf hoặc một phần tên")
    ap.add_argument("--cpu", action="store_true", help="chạy model hoàn toàn bằng CPU (ngụ ý --local, không dùng GPU)")
    ap.add_argument("--threads", type=int, default=0, help="số luồng CPU khi dùng --cpu (mặc định: 8)")
    ap.add_argument("--ctx", type=int, default=None,
                    help="context (mặc định: tự đọc từ server; --local: 16384). Làm ngân sách dọn context")
    ap.add_argument("--port", type=int, default=8081, help="cổng khi dùng --local")
    ap.add_argument("--effort", choices=["low", "medium", "high"], default="medium",
                    help="mức suy nghĩ: low ngắn gọn · medium (mặc định) · high suy nghĩ kỹ, tự kiểm tra")
    ap.add_argument("--think", choices=["on", "off", "auto"], default="auto",
                    help="bật/tắt chế độ suy nghĩ khi dùng --local")
    ap.add_argument("--embed", metavar="MODEL", help="model embedding .gguf (đường dẫn hoặc một phần tên); mặc định tự tìm")
    ap.add_argument("--no-embed", action="store_true", help="không dùng embedding (chỉ tìm theo từ khoá BM25)")
    ap.add_argument("--no-memory", action="store_true", help="tắt bộ nhớ ngoài")
    ap.add_argument("--no-summary", action="store_true", help="không tóm tắt hội thoại cũ khi đẩy ra khỏi context")
    ap.add_argument("--no-sandbox", action="store_true", help="KHÔNG nhốt bash trong workspace (nguy hiểm)")
    ap.add_argument("--verify", metavar="LỆNH", help="lệnh kiểm tra tự động chạy khi agent báo xong (vd: 'python3 -m unittest'); "
                    "thất bại thì agent phải sửa tiếp, không cho báo xong")
    ap.add_argument("--verify-rounds", type=int, default=3, help="số vòng sửa tối đa sau khi kiểm tra thất bại")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-steps", type=int, default=25)
    return ap


@contextlib.contextmanager
def backends(a):
    """Bật llama-server (nếu --local/--cpu) và server embedding; tự tắt khi thoát. Đặt a.host, a.model, a.embedder."""
    servers = []
    a.embedder = None
    try:
        if a.cpu and a.local is None:
            a.local = ""
        if a.local is not None:
            server = LocalServer(find_model(a.local or None), a.port, a.ctx or DEFAULT_CTX, reasoning=a.think,
                                 effort=a.effort, cpu=a.cpu, threads=(a.threads or 8) if a.cpu else 0)
            print(f"Khởi động llama-server ({'CPU' if a.cpu else 'GPU'}) · {server.model.name} · ctx {server.ctx}"
                  " ... (nạp model có thể mất 10–60s)", flush=True)
            servers.append(server)
            server.start()
            a.host, a.model = server.host, a.model or "local"
        if not a.no_memory and not a.no_embed:
            try:
                emb = EmbedServer(find_embed_model(a.embed), a.port + 1)
                print(f"Khởi động embedding (CPU) · {emb.model.name} ...", flush=True)
                servers.append(emb)
                emb.start()
                a.embedder = EmbedClient(emb.host)
            except ServeError as e:
                print(f"⚠ không bật embedding ({str(e).splitlines()[0]}); dùng tìm kiếm từ khoá.", file=sys.stderr)
        yield a
    finally:
        for srv in servers:
            srv.stop()


def make_client(a) -> tuple[LMStudioClient, int]:
    """Client + context thực tế (đọc từ server, hoặc --ctx, hoặc mặc định)."""
    if not a.host.startswith("http"):
        a.host = "http://" + a.host
    client = LMStudioClient(a.model, a.host, a.temperature)
    client.ensure_model()
    return client, a.ctx or client.context_length() or DEFAULT_CTX


def make_agent(a, client, ctx: int, ws: Workspace, ui: UI, memory: Memory | None, extra: str = "",
               mcp: bool = True) -> tuple[Agent, list]:
    tools = build_tools(ws, sandbox=not getattr(a, "no_sandbox", False))
    servers = []
    if mcp:
        mcp_tools, servers = load_mcp_tools(ws.root, warn=lambda m: ui.w(ui.c("warn", m)))
        tools.update(mcp_tools)
    def verify():
        out = tools["bash"].fn(a.verify)
        m = re.search(r"\[exit (\d+)\]\s*$", out)
        return bool(m and m.group(1) == "0"), out

    agent = Agent(client, tools, ws, ui=ui, memory=memory, effort=a.effort, text_tools=a.text_tools,
                  verify=verify if getattr(a, "verify", None) else None, verify_rounds=getattr(a, "verify_rounds", 3),
                  system_extra=extra, max_steps=a.max_steps, summarize=not a.no_summary,
                  max_context_tokens=int(ctx * CTX_USABLE), ctx_total=ctx,
                  auto_approve=not getattr(a, "ask", False))
    if memory:
        tools.update(memory_tools(memory, lambda: agent.turn))
    d = delegate_tool(agent)
    tools[d.name] = d
    agent.reset()  # nạp danh sách tool đầy đủ (cần cho --text-tools)
    return agent, servers


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "eval":
        from .evals import main as eval_main
        return eval_main(argv[1:])
    ap = argparse.ArgumentParser(prog="harness", parents=[common_parser()],
                                 description="Agent coding CLI cho LLM local (LM Studio / llama-server)")
    ap.add_argument("-C", "--cwd", default=".", help="workspace (mặc định: thư mục hiện tại)")
    ap.add_argument("-p", "--prompt", help="chạy một lần rồi thoát")
    ap.add_argument("--ask", action="store_true", help="hỏi duyệt trước khi ghi/xoá/chạy lệnh (mặc định: toàn quyền)")
    ap.add_argument("-y", "--yes", action="store_true", help=argparse.SUPPRESS)  # giữ cho tương thích
    ap.add_argument("--system", help="prompt bổ sung (văn bản, hoặc @đường/dẫn/file)")
    ap.add_argument("--goal", help="mục tiêu lớn: agent tự lập kế hoạch, harness giao từng việc nhỏ với ngân sách bước riêng")
    ap.add_argument("--step-budget", type=int, default=30, help="số bước tối đa cho MỖI việc trong --goal")
    ap.add_argument("--max-items", type=int, default=10, help="số việc tối đa trong --goal")
    ap.add_argument("--no-require-tests", action="store_true", help="--goal: không đòi test đi kèm khi đổi mã")
    ap.add_argument("--oracle", metavar="SCRIPT", help="script nghiệm thu độc lập (ngoài sandbox) cho --stages")
    ap.add_argument("--stages", action="store_true", help="làm lần lượt từng giai đoạn của --oracle, mỗi giai đoạn kiểm chứng bằng oracle")
    ap.add_argument("--start-stage", type=int, default=1, help="bắt đầu từ giai đoạn này")
    ap.add_argument("--stage-rounds", type=int, default=4, help="số vòng sửa tối đa cho mỗi giai đoạn")
    ap.add_argument("--resume", nargs="?", const="last", metavar="TÊN", help="mở lại phiên đã lưu (mặc định: phiên gần nhất)")
    ap.add_argument("--session", metavar="TÊN", help="lưu phiên này thêm dưới tên TÊN")
    a = ap.parse_args(argv)
    try:
        with backends(a):
            return run(a)
    except (LLMError, ServeError) as e:
        print(f"✗ {e}", file=sys.stderr)
        return 1


def run(a) -> int:
    ws = Workspace(a.cwd)
    ui = UI()
    client, ctx = make_client(a)
    memory = None if a.no_memory else Memory(ws.root / ".harness" / "memory.db", uuid.uuid4().hex[:8], a.embedder)
    extra = a.system or ""
    if extra.startswith("@"):
        extra = open(extra[1:]).read()
    if memory and memory.embedder:
        n = memory.backfill()
        if n:
            ui.w(ui.c("muted", f"◌ đã nhúng bù {n} mục cũ vào bộ nhớ"))
    agent, mcp_servers = make_agent(a, client, ctx, ws, ui, memory, extra)
    sess_dir = ws.root / ".harness" / "sessions"
    save = lambda: [agent.save(sess_dir / f"{n}.json") for n in ("last", a.session) if n]

    def save_quietly():
        try:
            save()
        except OSError:
            pass   # không để lỗi ghi đĩa che mất lỗi gốc
    try:
        if a.resume:
            if agent.load(sess_dir / f"{a.resume}.json"):
                ui.w(ui.c("muted", f"◌ đã mở lại phiên '{a.resume}' ({agent.turn} lượt)"))
            else:
                ui.error(f"không tìm thấy phiên '{a.resume}' trong {sess_dir}")
        if a.stages:
            from .stages import Oracle, exit_code, report as stage_report, run_stages
            if not a.oracle:
                ui.error("--stages cần --oracle SCRIPT")
                return 1
            unit = None
            if a.verify:
                def unit():
                    out = agent.tools["bash"].fn(a.verify)
                    m = re.search(r"\[exit (\d+)\]\s*$", out)
                    return bool(m and m.group(1) == "0"), out
            oracle = Oracle(a.oracle, ws.root)
            last_stage = max((s["id"] for s in oracle.list_stages()), default=0)
            results = run_stages(agent, oracle, a.step_budget, a.stage_rounds, a.start_stage,
                                 unit_verify=unit, on_stage_done=save)
            save()
            ui.w("\n" + stage_report(results))
            return exit_code(results, last_stage)
        if a.goal:
            from .goal import report, run_goal
            results = run_goal(agent, a.goal, a.step_budget, a.max_items, require_tests=not a.no_require_tests,
                               on_item_done=save)
            save()
            ui.w("\n" + report(results))
            return 0 if results and all(r.status == "done" for r in results) else 2
        if a.prompt:
            agent.run(a.prompt)
            save()
            return 0
        ui.banner(client.model, str(ws.root))
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
            if line == "/context":
                ui.w(ui.c("muted", f"context {agent.ctx_info() or 'chưa đo'} · ngân sách hội thoại ~{agent._budget()} ký tự "
                                   f"(~{agent._cpt:.1f} ký tự/token) · tóm tắt {'có' if agent.summary else 'chưa'}"))
                if agent.summary:
                    ui.w(ui.c("dim", agent.summary))
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
                save()
            except KeyboardInterrupt:
                ui.w(ui.c("muted", "\n[đã ngắt]"))
            except LLMError as e:   # server lỗi giữa chừng: báo rồi cho gõ tiếp, đừng đóng cả phiên
                ui.error(str(e))
                if agent.drop_dangling_turn():
                    ui.w(ui.c("muted", "(lượt này chưa được xử lý, hãy gõ lại)"))
                save_quietly()
    except LLMError as e:
        ui.error(str(e))
        save_quietly()   # giữ lại những gì đã làm để --resume
        return 1
    finally:
        for s in mcp_servers:
            s.stop()


if __name__ == "__main__":
    sys.exit(main())

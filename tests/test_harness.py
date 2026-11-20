import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.agent import Agent, parse_text_tool_calls
from harness.client import Reply
from harness.ui import UI
from harness.tools import Workspace, build_tools


class FakeClient:
    """Trả lời theo kịch bản: lần 1 gọi tool, lần 2 trả lời cuối."""
    def __init__(self, script):
        self.script = list(script)

    def chat(self, messages, tools=None, on_token=None, on_thinking=None):
        return self.script.pop(0)


class Tests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.ws = Workspace(self.d.name)
        self.tools = build_tools(self.ws)

    def test_path_escape_blocked(self):
        with self.assertRaises(ValueError):
            self.ws.path("../etc/passwd")

    def test_edit_requires_unique(self):
        self.tools["write_file"].fn("a.txt", "x x")
        self.assertIn("2 lần", self.tools["edit_file"].fn("a.txt", "x", "y"))
        self.assertIn("Đã sửa", self.tools["edit_file"].fn("a.txt", "x x", "z"))

    def test_text_tool_parse(self):
        t = 'ok <tool_call>{"name":"bash","arguments":{"command":"ls"}}</tool_call>'
        self.assertEqual(parse_text_tool_calls(t), [{"id": "text_0", "name": "bash", "arguments": {"command": "ls"}}])

    def test_agent_loop_and_denial(self):
        client = FakeClient([
            Reply(tool_calls=[{"id": "c1", "name": "write_file", "arguments": {"path": "f.txt", "content": "hi"}}]),
            Reply(content="xong"),
        ])
        agent = Agent(client, self.tools, self.ws, approve=lambda n, a: False, ui=UI(io.StringIO()))
        self.assertEqual(agent.run("tạo file"), "xong")
        self.assertIn("từ chối", agent.messages[-2]["content"])
        self.assertFalse((self.ws.root / "f.txt").exists())

    def test_agent_approved(self):
        client = FakeClient([
            Reply(tool_calls=[{"id": "c1", "name": "write_file", "arguments": {"path": "f.txt", "content": "hi"}}]),
            Reply(content="xong"),
        ])
        Agent(client, self.tools, self.ws, auto_approve=True, ui=UI(io.StringIO())).run("tạo file")
        self.assertEqual((self.ws.root / "f.txt").read_text(), "hi")


class MoreTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.ws = Workspace(self.d.name)
        self.tools = build_tools(self.ws)

    def test_validation_missing_and_coerce(self):
        t = self.tools["read_file"]
        self.assertIn("thiếu tham số path", t.clean_args({}))
        self.assertEqual(t.clean_args({"path": "a", "limit": "5", "junk": 1}), {"path": "a", "limit": 5})
        self.assertIn("phải là integer", t.clean_args({"path": "a", "limit": "x"}))

    def test_glob_grep_todo(self):
        self.tools["write_file"].fn("src/a.py", "def foo():\n    pass\n")
        self.tools["write_file"].fn("b.txt", "foo")
        self.assertEqual(self.tools["glob"].fn("*.py"), "src/a.py")
        self.assertIn("src/a.py:1", self.tools["grep"].fn("foo", include="*.py"))
        self.assertNotIn("b.txt", self.tools["grep"].fn("foo", include="*.py"))
        self.assertIn("1/2", self.tools["todo"].fn("[x] a\n[ ] b"))

    def test_move_delete(self):
        self.tools["write_file"].fn("x.txt", "1")
        self.tools["move_file"].fn("x.txt", "d/y.txt")
        self.assertTrue((self.ws.root / "d/y.txt").exists())
        self.tools["delete_file"].fn("d")
        self.assertFalse((self.ws.root / "d").exists())
        self.assertIn("LỖI", self.tools["delete_file"].fn("."))

    def test_compact_and_loop_guard(self):
        agent = Agent(FakeClient([]), self.tools, self.ws, max_context_chars=500, ui=UI(io.StringIO()))
        for i in range(10):
            agent.messages.append({"role": "tool", "tool_call_id": str(i), "content": "x" * 400})
        self.assertGreater(agent.compact(), 0)
        self.assertIn("đã lược", agent.messages[1]["content"])
        calls = [Reply(tool_calls=[{"id": str(i), "name": "list_dir", "arguments": {}}]) for i in range(5)]
        agent2 = Agent(FakeClient(calls + [Reply(content="ok")]), self.tools, self.ws, ui=UI(io.StringIO()))
        agent2.run("x")
        self.assertTrue(any("4 lần" in (m.get("content") or "") for m in agent2.messages))

    def test_project_context_file(self):
        (self.ws.root / "HARNESS.md").write_text("QUY-TAC-RIENG")
        self.assertIn("QUY-TAC-RIENG", Agent(FakeClient([]), self.tools, self.ws, ui=UI(io.StringIO())).messages[0]["content"])


class ServeTests(unittest.TestCase):
    def test_command_flags(self):
        from pathlib import Path
        from harness.serve import LocalServer
        cmd = LocalServer(Path("/m.gguf"), port=9, ctx=4096).command("/bin/llama-server")
        for flag in ("--jinja", "-fa", "-ngl", "--no-webui"):
            self.assertIn(flag, cmd)
        self.assertEqual(cmd[cmd.index("-c") + 1], "4096")


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.ws = Workspace(self.d.name)
        from harness.memory import Memory, memory_tools
        self.mem = Memory(":memory:", "s1")
        self.addCleanup(self.mem.close)
        self.tools = build_tools(self.ws)
        self.agent = None
        self.memory_tools = memory_tools

    def make(self, script, **kw):
        a = Agent(FakeClient(script), self.tools, self.ws, ui=UI(io.StringIO()), memory=self.mem, **kw)
        self.tools.update(self.memory_tools(self.mem, lambda: a.turn))
        return a

    def test_diacritics_and_live_exclusion(self):
        self.mem.add("Sửa lỗi hàm cộng trong calc.py", "turn", 3)
        self.assertEqual(len(self.mem.search("ham cong")), 1)
        self.assertEqual(self.mem.search("ham cong", first_live_turn=3), [])   # còn trong context
        self.assertEqual(len(self.mem.search("ham cong", first_live_turn=4)), 1)  # đã đẩy ra ngoài

    def test_dedupe(self):
        self.assertEqual(self.mem.add("abc def", "note"), 1)
        self.assertEqual(self.mem.add("abc def", "note"), 0)

    def test_turn_saved_and_eviction(self):
        a = self.make([Reply(content=f"đáp {i} " + "x" * 300) for i in range(6)], max_context_chars=900)
        for i in range(6):
            a.run(f"câu hỏi số {i} về chủ đề{i}")
        users = [m for m in a.messages if "_turn" in m]
        self.assertLessEqual(len(users), 4)           # lượt cũ đã bị đẩy ra
        self.assertEqual(a.messages[1]["role"], "user")  # lịch sử vẫn hợp lệ, bắt đầu bằng user
        self.assertGreater(a.first_live_turn, 1)
        self.assertEqual(self.mem.stats()["turn"], 6)    # nhưng tất cả vẫn nằm trong bộ nhớ

    def test_recall_injected_ephemeral(self):
        a = self.make([Reply(content="a1"), Reply(content="a2")], max_context_chars=300)
        a.run("hãy nhớ mật khẩu wifi là ZEBRA42")
        a.first_live_turn = 2  # giả lập lượt 1 đã bị đẩy ra khỏi context
        a.run("mật khẩu wifi là gì")
        sent_view = a._view()
        self.assertIn("ZEBRA42", [m for m in sent_view if m["role"] == "user"][-1]["content"])  # model được mớm
        self.assertNotIn("ZEBRA42", [m for m in a.messages if m.get("_turn") == 2][0]["content"])  # lịch sử sạch
        self.assertFalse(any(k.startswith("_") for m in sent_view for k in m))  # khoá nội bộ không bị gửi đi

    def test_remember_recall_tools(self):
        a = self.make([])
        self.assertIn("Đã lưu", self.tools["remember"].fn("Dự án dùng pytest"))
        self.assertIn("pytest", self.tools["recall"].fn("pytest"))
        self.assertIn("Không tìm thấy", self.tools["recall"].fn("không-có-gì-cả"))


class MemoryOrderTest(unittest.TestCase):
    def test_recall_after_compaction_same_turn(self):
        """Lượt cũ bị đẩy ra ngay khi lượt mới bắt đầu thì vẫn phải được gợi nhớ lượt đó."""
        from harness.memory import Memory
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup); ws = Workspace(d.name)
        mem = Memory(":memory:", "s"); self.addCleanup(mem.close)
        a = Agent(FakeClient([Reply(content="x" * 200)] * 5), build_tools(ws), ws, ui=UI(io.StringIO()),
                  memory=mem, max_context_chars=1300)
        for q in ["mật khẩu wifi là ZEBRA42", "thơ thu", "thơ biển", "thơ núi", "mật khẩu wifi là gì"]:
            a.run(q)
        self.assertIn("ZEBRA42", a._recalled)


class CpuServeTest(unittest.TestCase):
    def test_cpu_flags(self):
        from pathlib import Path
        from harness.serve import LocalServer
        cmd = LocalServer(Path("/m.gguf"), cpu=True, threads=6).command("/b")
        self.assertEqual(cmd[cmd.index("-ngl") + 1], "0")
        self.assertEqual(cmd[cmd.index("-t") + 1], "6")


class FakeEmbedder:
    """Embedding giả theo chủ đề: 'mật khẩu'/'đăng nhập' cùng hướng, nên hai cách nói khác nhau vẫn khớp."""
    TOPICS = [("mat", "dang", "password"), ("tho",), ("thoi", "troi")]

    def embed(self, texts):
        import array, math, unicodedata
        out = []
        for t in texts:
            t = "".join(c for c in unicodedata.normalize("NFD", t.lower().replace("đ", "d")) if unicodedata.category(c) != "Mn")
            words = set(t.split())
            v = [1.0 if words & set(topic) else 0.0 for topic in self.TOPICS]
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append(array.array("f", [x / n for x in v]))
        return out


class HybridTests(unittest.TestCase):
    def test_semantic_hit_without_shared_words(self):
        from harness.memory import Memory
        m = Memory(":memory:", "s", FakeEmbedder()); self.addCleanup(m.close)
        m.add("Mật khẩu wifi là ZEBRA42", "turn", 1)
        m.add("Bài thơ mùa thu rất hay", "turn", 2)
        self.assertEqual(m.search("thông tin đăng nhập", k=1)[0].content, "Mật khẩu wifi là ZEBRA42")  # BM25 một mình sẽ trượt
        m2 = Memory(":memory:", "s"); self.addCleanup(m2.close)
        self.assertEqual(m2.search("thông tin đăng nhập"), [])

    def test_embed_failure_falls_back_to_bm25(self):
        from harness.embed import EmbedError
        from harness.memory import Memory

        class Broken:
            def embed(self, texts):
                raise EmbedError("down")
        m = Memory(":memory:", "s", Broken()); self.addCleanup(m.close)
        m.add("Mật khẩu wifi là ZEBRA42", "turn", 1)
        self.assertEqual(len(m.search("mat khau")), 1)
        self.assertEqual(m.stats().get("vector"), 0)

    def test_backfill(self):
        from harness.memory import Memory
        m = Memory(":memory:", "s"); self.addCleanup(m.close)
        m.add("Mật khẩu wifi", "turn", 1)
        m.embedder = FakeEmbedder()
        self.assertEqual(m.backfill(), 1)
        self.assertEqual(m.stats()["vector"], 1)


class LexicalFilterTests(unittest.TestCase):
    def test_lexically_strong(self):
        from harness.memory import lexically_strong as ls
        self.assertFalse(ls("Hàm add đang trừ thay vì cộng", "thông tin đăng nhập"))   # đăng ≠ đang
        self.assertTrue(ls("Mật khẩu wifi là ZEBRA42", "zebra42 là gì"))                # mã định danh
        self.assertTrue(ls("Mật khẩu wifi", "mat khau"))                                # gõ không dấu → so sánh bỏ dấu
        self.assertFalse(ls("Mật khẩu wifi", "hôm nay là gì"))                          # chỉ có từ phổ biến


class EffortTests(unittest.TestCase):
    def test_effort_changes_prompt_and_server_flag(self):
        from pathlib import Path
        from harness.serve import LocalServer
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup); ws = Workspace(d.name)
        mk = lambda e: Agent(FakeClient([]), build_tools(ws), ws, ui=UI(io.StringIO()), effort=e).messages[0]["content"]
        self.assertIn("ngắn gọn", mk("low"))
        self.assertIn("tự kiểm tra lại", mk("high"))
        self.assertNotIn("{effort_rule}", mk("medium"))
        cmd = LocalServer(Path("/m"), effort="high").command("/b")
        self.assertEqual(cmd[cmd.index("--reasoning-effort") + 1], "high")


class WebSearchParseTest(unittest.TestCase):
    def test_parse(self):
        from harness.tools import parse_search_results
        page = ('<div class="result results_links"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fa.com%2Fx&amp;rut=1">Tiêu &amp; đề</a>'
                '<a class="result__snippet" href="x">Đoạn <b>trích</b></a></div>'
                '<div class="result results_links"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fb.org">B</a></div>')
        out = parse_search_results(page)
        self.assertIn("1. Tiêu & đề\n   https://a.com/x\n   Đoạn trích", out)
        self.assertIn("2. B\n   https://b.org", out)


class MCPTests(unittest.TestCase):
    def test_load_and_call(self):
        import json, sys
        from pathlib import Path
        from harness.mcp import load_mcp_tools
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup)
        (Path(d.name) / ".harness").mkdir()
        (Path(d.name) / ".harness" / "mcp.json").write_text(json.dumps(
            {"mcpServers": {"demo": {"command": sys.executable, "args": [str(Path(__file__).parent / "fake_mcp.py")]},
                            "hong": {"command": "/không/có/lệnh/này"}}}))
        warns = []
        tools, servers = load_mcp_tools(Path(d.name), warn=warns.append)
        try:
            self.assertEqual(set(tools), {"demo__echo", "demo__boom"})
            self.assertEqual(len(warns), 1)  # server hỏng bị bỏ qua, không làm sập
            t = tools["demo__echo"]
            self.assertEqual(t.clean_args({"text": "ab", "times": "2"}), {"text": "ab", "times": 2})
            self.assertEqual(t.fn(text="ab", times=2), "abab")
            self.assertEqual(t.schema()["function"]["parameters"]["required"], ["text"])
            self.assertTrue(tools["demo__boom"].fn().startswith("LỖI: hỏng rồi"))
        finally:
            for s in servers:
                s.stop()


class DelegateTest(unittest.TestCase):
    def test_delegate_returns_only_conclusion(self):
        from harness.agent import delegate_tool
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup); ws = Workspace(d.name)
        (ws.root / "a.py").write_text("x = 1\n")
        tools = build_tools(ws)
        client = FakeClient([Reply(tool_calls=[{"id": "1", "name": "read_file", "arguments": {"path": "a.py"}}]),
                             Reply(content="Kết luận: a.py đặt x = 1")])
        parent = Agent(client, tools, ws, ui=UI(io.StringIO()))
        out = delegate_tool(parent).fn(task="a.py làm gì?")
        self.assertEqual(out, "Kết luận: a.py đặt x = 1")


class SmartClient:
    """Trả lời tóm tắt khi bị gọi để tóm tắt; còn lại theo kịch bản."""
    def __init__(self, script, prompt_tokens=0, fail_first=None):
        self.script, self.prompt_tokens, self.fail_first, self.summaries = list(script), prompt_tokens, fail_first, 0

    def chat(self, messages, tools=None, on_token=None, on_thinking=None):
        from harness.agent import SUMMARY_SYSTEM
        if messages[0]["content"] == SUMMARY_SYSTEM:
            self.summaries += 1
            return Reply(content="- người dùng đã hỏi vài câu; mật khẩu là ZEBRA42")
        if self.fail_first:
            e, self.fail_first = self.fail_first, None
            raise e
        r = self.script.pop(0)
        r.prompt_tokens = self.prompt_tokens
        return r


class ContextFeatureTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.ws = Workspace(self.d.name)

    def agent(self, client, **kw):
        return Agent(client, build_tools(self.ws), self.ws, ui=UI(io.StringIO()), **kw)

    def test_calibrate_and_ctx_info(self):
        a = self.agent(SmartClient([Reply(content="ok")], prompt_tokens=1500), max_context_tokens=10000, ctx_total=16000)
        a.run("xin chào")
        self.assertEqual(a.last_prompt_tokens, 1500)
        self.assertEqual(a.ctx_info(), "1.5k/16.0k")
        self.assertNotEqual(a._cpt, 3.0)  # đã học lại tỉ lệ ký tự/token từ server

    def test_summary_on_eviction(self):
        from harness.memory import Memory
        c = SmartClient([Reply(content="x" * 300)] * 6)
        mem = Memory(":memory:", "s"); self.addCleanup(mem.close)
        a = self.agent(c, memory=mem, summarize=True, max_context_chars=1300)
        for q in ["mật khẩu là ZEBRA42", "chuyện 2", "chuyện 3", "chuyện 4"]:
            a.run(q)
        self.assertGreater(c.summaries, 0)
        self.assertIn("ZEBRA42", a.summary)
        self.assertIn("ZEBRA42", a.messages[0]["content"])   # bản tóm tắt nằm trong prompt hệ thống

    def test_summary_works_without_memory(self):
        c = SmartClient([Reply(content="x" * 300)] * 6)
        a = self.agent(c, summarize=True, max_context_chars=1300)
        for q in ["a", "b", "c", "d"]:
            a.run(q)
        self.assertGreater(c.summaries, 0)

    def test_overflow_retry(self):
        from harness.client import LLMError
        c = SmartClient([Reply(content="xong")], fail_first=LLMError("HTTP 400: context size exceeded"))
        self.assertEqual(self.agent(c).run("x"), "xong")
        c2 = SmartClient([], fail_first=LLMError("HTTP 500: lỗi khác"))
        with self.assertRaises(LLMError):
            self.agent(c2).run("x")

    def test_save_load_roundtrip(self):
        from harness.memory import Memory
        m1 = Memory(":memory:", "phien-A"); self.addCleanup(m1.close)
        a = self.agent(SmartClient([Reply(content="đáp 1")]), memory=m1)
        a.run("câu 1")
        a.summary = "- tóm tắt"
        a.save(self.ws.root / "s.json")
        m2 = Memory(":memory:", "phien-B"); self.addCleanup(m2.close)
        b = self.agent(SmartClient([]), memory=m2)
        self.assertTrue(b.load(self.ws.root / "s.json"))
        self.assertEqual((b.turn, b.summary, m2.session), (1, "- tóm tắt", "phien-A"))
        self.assertIn("tóm tắt", b.messages[0]["content"])
        self.assertFalse(b.load(self.ws.root / "khong-co.json"))

    def test_full_autonomy_by_default_and_ask_mode(self):
        call = Reply(tool_calls=[{"id": "1", "name": "write_file", "arguments": {"path": "f.txt", "content": "hi"}}])
        self.agent(SmartClient([call, Reply(content="xong")])).run("tạo")
        self.assertTrue((self.ws.root / "f.txt").exists())          # không hỏi duyệt
        (self.ws.root / "f.txt").unlink()
        a = self.agent(SmartClient([call, Reply(content="xong")]), auto_approve=False, approve=lambda n, x: False)
        a.run("tạo")
        self.assertFalse((self.ws.root / "f.txt").exists())          # --ask thì vẫn chặn được


class EvalRunnerTests(unittest.TestCase):
    def ns(self):
        import argparse
        return argparse.Namespace(effort="medium", text_tools=False, max_steps=10, no_summary=True, ask=False,
                                  embedder=None, no_memory=True)

    def test_run_task_pass_and_fail(self):
        from harness.evals import TASKS, run_task
        fix = next(t for t in TASKS if t.name == "fix_bug")
        edit = Reply(tool_calls=[{"id": "1", "name": "edit_file",
                                  "arguments": {"path": "calc.py", "old": "a - b", "new": "a + b"}}])
        r = run_task(self.ns(), SmartClient([edit, Reply(content="đã sửa")]), 16000, fix)
        self.assertTrue(r.passed, r.detail)
        self.assertEqual(r.tool_calls, 1)
        r2 = run_task(self.ns(), SmartClient([Reply(content="không làm gì")]), 16000, fix)
        self.assertFalse(r2.passed)

    def test_checks(self):
        from harness.evals import _count_callers, _not_found, _precise_edit
        self.assertTrue(_not_found(None, ["File không tồn tại."])[0])
        self.assertFalse(_not_found(None, ["Nội dung là: hello"])[0])
        self.assertTrue(_count_callers(None, ["Có 2 file."])[0])
        self.assertFalse(_count_callers(None, ["Có 3 file."])[0])
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup)
        from pathlib import Path
        (Path(d.name) / "settings.ini").write_text("[server]\nhost = 0.0.0.0\nport = 9090\ndebug = false\n")
        self.assertTrue(_precise_edit(Path(d.name), [])[0])

    def test_compare(self):
        import json
        from pathlib import Path
        from harness.evals import compare
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup)
        f = Path(d.name) / "old.json"
        f.write_text(json.dumps({"label": "cũ", "results": [{"task": "t", "passed": False}, {"task": "t", "passed": True}]}))
        out = compare(str(f), [{"task": "t", "passed": True}, {"task": "t", "passed": True}])
        self.assertIn("1/2 → 2/2 ▲", out)


class SearchBackendTests(unittest.TestCase):
    def test_format_searx(self):
        from harness.tools import format_searx
        out = format_searx({"results": [{"title": "A", "url": "https://a.io", "content": "mô tả"}, {"title": "B", "url": "https://b.io"}]})
        self.assertEqual(out, "1. A\n   https://a.io\n   mô tả\n2. B\n   https://b.io")


@unittest.skipUnless(__import__("harness.sandbox", fromlist=["x"]).available(), "kernel không có Landlock")
class SandboxTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.ws = Workspace(self.d.name)
        self.bash = build_tools(self.ws)["bash"].fn

    def test_works_inside_workspace(self):
        out = self.bash("mkdir sub && echo xin chào > sub/f.txt && cat sub/f.txt && python3 -c 'print(6*7)'")
        self.assertIn("xin chào", out)
        self.assertIn("42", out)
        self.assertTrue((self.ws.root / "sub" / "f.txt").exists())

    def test_cannot_read_outside(self):
        self.assertNotIn("root:", self.bash("cat /etc/passwd"))
        self.assertIn("denied", self.bash("ls /home").lower().replace("permission denied", "denied"))
        self.assertNotIn("root:", self.bash("ln -s /etc/passwd lk && cat lk"))  # đường vòng qua symlink

    def test_cannot_write_outside(self):
        marker = self.ws.root.parent / "harness-escape-marker.txt"
        self.bash(f"echo x > {marker}")
        self.bash("echo x > ../harness-escape-marker2.txt")
        self.bash("echo x > /tmp/harness-escape-marker3.txt")
        self.assertFalse(marker.exists())
        self.assertFalse((self.ws.root.parent / "harness-escape-marker2.txt").exists())
        self.assertFalse(Path("/tmp/harness-escape-marker3.txt").exists())

    def test_cannot_cd_out(self):
        self.assertIn("denied", self.bash("cd .. && ls").lower())

    def test_fail_closed_when_unavailable(self):
        import harness.tools as T
        real = T._sandbox.available
        T._sandbox.available = lambda: False
        try:
            out = build_tools(self.ws)["bash"].fn("echo hi")
        finally:
            T._sandbox.available = real
        self.assertIn("LỖI", out)
        self.assertNotIn("hi\n", out)
        self.assertEqual(build_tools(self.ws, sandbox=False)["bash"].fn("echo hi").split()[0], "hi")


@unittest.skipUnless(__import__("harness.sandbox", fromlist=["x"]).available(), "kernel không có Landlock")
class SandboxHintTests(unittest.TestCase):
    def test_denied_output_explains_and_prompt_says_relative(self):
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup); ws = Workspace(d.name)
        out = build_tools(ws)["bash"].fn("cat /etc/passwd")
        self.assertIn("[sandbox]", out)
        self.assertIn("TƯƠNG ĐỐI", out)
        self.assertNotIn("[sandbox]", build_tools(ws)["bash"].fn("echo ok"))
        a = Agent(FakeClient([]), build_tools(ws), ws, ui=UI(io.StringIO()))
        self.assertIn("đường dẫn TƯƠNG ĐỐI", a.messages[0]["content"])


class PromptHidesAbsolutePathTest(unittest.TestCase):
    def test_no_absolute_root_in_system_prompt(self):
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup); ws = Workspace(d.name)
        sys_prompt = Agent(FakeClient([]), build_tools(ws), ws, ui=UI(io.StringIO())).messages[0]["content"]
        self.assertNotIn(str(ws.root), sys_prompt)   # model hay gõ sai đường dẫn tuyệt đối
        self.assertIn("thư mục hiện tại", sys_prompt)


class LongTurnCompactionTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory(); self.addCleanup(self.d.cleanup); self.ws = Workspace(self.d.name)

    def agent(self, **kw):
        return Agent(SmartClient([]), build_tools(self.ws), self.ws, ui=UI(io.StringIO()), **kw)

    def test_over_budget_keeps_only_current_turn(self):
        a = self.agent(summarize=True, max_context_chars=900)
        a.messages += [{"role": "user", "content": "câu cũ " + "x" * 600, "_turn": 1}, {"role": "assistant", "content": "y" * 600},
                       {"role": "user", "content": "câu mới", "_turn": 2}, {"role": "assistant", "content": "ok"}]
        a.compact()
        self.assertEqual([m["_turn"] for m in a.messages if "_turn" in m], [2])  # trước đây luôn giữ 2 lượt nên không dọn được

    def test_single_long_turn_drops_old_steps_keeps_pairs(self):
        a = self.agent(summarize=True, max_context_chars=3000)
        a.turn = 1
        a.messages.append({"role": "user", "content": "làm app", "_turn": 1})
        for i in range(12):
            a.messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": str(i), "type": "function",
                               "function": {"name": "read_file", "arguments": "{}"}}]})
            a.messages.append({"role": "tool", "tool_call_id": str(i), "content": f"kết quả {i} " + "z" * 700})
        before = len(a.messages)
        a.compact()
        self.assertLess(len(a.messages), before)
        self.assertEqual(a.messages[1]["content"], "làm app")               # tin nhắn của lượt được giữ
        self.assertEqual(a.messages[2]["role"], "assistant")                # không bắt đầu bằng tool mồ côi
        for i, m in enumerate(a.messages):
            if m["role"] == "tool":
                self.assertEqual(a.messages[i - 1]["role"] in ("assistant", "tool"), True)
        self.assertIn("kết quả 11", a.messages[-1]["content"])              # bước mới nhất còn nguyên


class VerifyGateTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory(); self.addCleanup(self.d.cleanup); self.ws = Workspace(self.d.name)

    def test_cannot_finish_until_verify_passes(self):
        results, calls = [(False, "FAILED: 1 test"), (True, "OK")], []
        def verify():
            calls.append(1)
            return results.pop(0)
        out = io.StringIO()
        a = Agent(SmartClient([Reply(content="xong"), Reply(content="đã sửa")]), build_tools(self.ws), self.ws,
                  ui=UI(out), verify=verify)
        self.assertEqual(a.run("làm"), "đã sửa")
        self.assertEqual(len(calls), 2)
        self.assertTrue(any("[Kiểm tra tự động]" in (m.get("content") or "") and "FAILED" in m["content"] for m in a.messages))
        self.assertIn("đạt", out.getvalue())

    def test_gives_up_honestly_after_rounds(self):
        out = io.StringIO()
        a = Agent(SmartClient([Reply(content="xong")] * 5), build_tools(self.ws), self.ws, ui=UI(out),
                  verify=lambda: (False, "vẫn lỗi"), verify_rounds=2)
        a.run("làm")
        self.assertIn("VẪN thất bại", out.getvalue())


class EditHintTests(unittest.TestCase):
    def test_hint_shows_closest_block(self):
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup); ws = Workspace(d.name)
        t = build_tools(ws)
        t["write_file"].fn("c.py", "def add(a, b):\n    return a - b\n\nprint(add(1, 2))\n")
        out = t["edit_file"].fn("c.py", "  return a -b", "  return a + b")
        self.assertIn("2\t    return a - b", out)       # chép lại đúng dòng thật, có số dòng
        self.assertIn("không tìm thấy", t["edit_file"].fn("c.py", "hoàn toàn khác biệt xyz 123", "q"))


class ToolCallArgsCountedTests(unittest.TestCase):
    def test_write_file_args_count_and_shrink(self):
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup); ws = Workspace(d.name)
        a = Agent(SmartClient([]), build_tools(ws), ws, ui=UI(io.StringIO()), max_context_chars=15000)
        big = json.dumps({"path": "db.py", "content": "x = 1\n" * 400})
        a.messages.append({"role": "user", "content": "viết db", "_turn": 1})
        for i in range(8):
            a.messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": str(i), "type": "function",
                               "function": {"name": "write_file", "arguments": big}}]})
            a.messages.append({"role": "tool", "tool_call_id": str(i), "content": "Đã tạo db.py"})
        self.assertGreater(a._size(), 16000)          # trước đây coi là ~100 ký tự
        a.compact()
        self.assertLess(a._size(), 11000)   # 6 tin gần nhất được giữ nguyên, phần cũ hơn đã lược
        self.assertIn("x = 1", a.messages[-2]["tool_calls"][0]["function"]["arguments"])  # bước mới nhất còn đủ nội dung
        old = json.loads(a.messages[2]["tool_calls"][0]["function"]["arguments"])   # vẫn là JSON hợp lệ
        self.assertIn("đã lược", old["content"])
        self.assertEqual(old["path"], "db.py")


class StaleFileAndNudgeTests(unittest.TestCase):
    def test_stale_file_hint(self):
        import os, time
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup); ws = Workspace(d.name); t = build_tools(ws)
        t["write_file"].fn("a.py", "x = 1\ny = 2\n")
        t["read_file"].fn("a.py")
        (ws.root / "a.py").write_text("x = 100\ny = 2\nz = 3\n")      # file bị đổi bởi nơi khác (vd bash sed)
        os.utime(ws.root / "a.py", (time.time() + 5, time.time() + 5))
        out = t["edit_file"].fn("a.py", "x = 1\n", "x = 9\n")
        self.assertIn("ĐÃ THAY ĐỔI", out)
        t["read_file"].fn("a.py")                                      # đọc lại thì không còn cảnh báo thừa
        self.assertNotIn("ĐÃ THAY ĐỔI", t["edit_file"].fn("a.py", "không có đoạn này", "q"))
        t["edit_file"].fn("a.py", "x = 100", "x = 7")                   # tự sửa của mình không bị coi là đã đổi
        self.assertNotIn("ĐÃ THAY ĐỔI", t["edit_file"].fn("a.py", "không có đoạn này", "q"))

    def test_nudge_todo_and_python_c(self):
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup); ws = Workspace(d.name)
        a = Agent(SmartClient([]), build_tools(ws), ws, ui=UI(io.StringIO()))
        st = {"nudged": 0, "todo": 0}
        self.assertIn("todo", a._nudge([], 25, st))
        self.assertEqual(a._nudge([], 30, st), "")                      # không nhắc dồn dập (cách ít nhất 12 bước)
        hist = [("bash", {"command": "python3 -c 'print(1)'"})] * 8
        st2 = {"nudged": 0, "todo": 30}
        self.assertIn("read_file", a._nudge(hist, 31, st2))
        self.assertEqual(a._nudge([("read_file", {})] * 8, 60, {"nudged": 0, "todo": 55, "edit": 50}), "")
        self.assertIn("chưa sửa được file nào", a._nudge([("read_file", {})] * 8, 60, {"nudged": 0, "todo": 55, "edit": 20}))


class GoalModeTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory(); self.addCleanup(self.d.cleanup); self.ws = Workspace(self.d.name)

    def agent(self, script, **kw):
        a = Agent(SmartClient(script), build_tools(self.ws), self.ws, ui=UI(io.StringIO()), **kw)
        return a

    def todo(self, text):
        return Reply(tool_calls=[{"id": "t", "name": "todo", "arguments": {"todos": text}}])

    def test_parse_plan(self):
        from harness.goal import parse_plan
        self.assertEqual(parse_plan(["[ ] a", "[x] b", "rác", "[!] c (bị chặn)"]), [(" ", "a"), ("x", "b"), ("!", "c (bị chặn)")])

    def test_runs_items_one_by_one(self):
        from harness.goal import run_goal
        a = self.agent([self.todo("[ ] làm A\n[ ] làm B"), Reply(content="đã lập kế hoạch"),   # giai đoạn lập kế hoạch
                        self.todo("[x] làm A\n[ ] làm B"), Reply(content="xong A"),             # việc 1: tự đánh [x]
                        Reply(content="xong B")])                                              # việc 2: kết thúc tự nhiên
        res = run_goal(a, "mục tiêu", step_budget=5, require_change=False)
        self.assertEqual([(r.index, r.status) for r in res], [(1, "done"), (2, "done")])
        self.assertTrue(all(line.startswith("[x]") for line in a.tools["todo"].state))
        self.assertEqual(a.max_steps, 25)   # ngân sách bước gốc được khôi phục

    def test_blocked_item_moves_on(self):
        from harness.goal import report, run_goal
        loop = [Reply(tool_calls=[{"id": str(i), "name": "list_dir", "arguments": {"path": "."}}]) for i in range(3)]
        a = self.agent([self.todo("[ ] việc khó\n[ ] việc dễ"), Reply(content="kế hoạch xong")]
                       + loop + [Reply(content="xong việc dễ")])
        res = run_goal(a, "mục tiêu", step_budget=3, plan_steps=5, require_change=False)
        self.assertEqual([r.status for r in res], ["blocked", "done"])   # hết bước ở việc 1 thì sang việc 2
        self.assertIn("[!] việc khó (bị chặn", a.tools["todo"].state[0])
        self.assertIn("1/2", report(res))

    def test_no_plan_is_reported(self):
        from harness.goal import run_goal
        a = self.agent([Reply(content="không làm gì")])
        self.assertEqual(run_goal(a, "x", step_budget=3, plan_steps=2, require_change=False), [])


class VerifyBaselineAndAdviceTests(unittest.TestCase):
    def test_fingerprint_and_regression(self):
        from harness.agent import _no_regression, fingerprint
        out1 = "FAIL: test_a (m.T.test_a)\nERROR: test_b (m.T.test_b)\nFAILED (failures=1, errors=1)"
        out2 = "FAIL: test_a (m.T.test_a)\nFAILED (failures=1)"
        out3 = "FAIL: test_c (m.T.test_c)\nFAILED (failures=1)"
        self.assertTrue(_no_regression(fingerprint(out2), fingerprint(out1)))    # ít lỗi hơn: không hồi quy
        self.assertFalse(_no_regression(fingerprint(out3), fingerprint(out1)))   # lỗi mới: hồi quy
        self.assertFalse(_no_regression(fingerprint(out1), None))

    def test_known_failure_does_not_block_and_new_one_does(self):
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup); ws = Workspace(d.name)
        a = Agent(SmartClient([Reply(content="xong")]), build_tools(ws), ws, ui=UI(io.StringIO()),
                  verify=lambda: (False, "FAIL: test_cu (m.T.test_cu)\nFAILED (failures=1)"))
        a.verify_baseline = frozenset({"test_cu (m.T.test_cu)"})
        self.assertEqual(a.run("việc"), "xong")                                  # không bị ép sửa lỗi cũ
        self.assertFalse(any("[Kiểm tra tự động]" in (m.get("content") or "") for m in a.messages))
        b = Agent(SmartClient([Reply(content="xong"), Reply(content="sửa rồi")]), build_tools(ws), ws,
                  ui=UI(io.StringIO()), verify_rounds=1,
                  verify=lambda: (False, "FAIL: test_moi (m.T.test_moi)\nFAILED (failures=1)"))
        b.verify_baseline = frozenset({"test_cu (m.T.test_cu)"})
        b.run("việc")
        self.assertTrue(any("[Kiểm tra tự động]" in (m.get("content") or "") for m in b.messages))  # lỗi mới thì chặn

    def test_repeat_failure_gets_advice(self):
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup); ws = Workspace(d.name)
        a = Agent(SmartClient([Reply(content="xong")] * 4), build_tools(ws), ws, ui=UI(io.StringIO()),
                  verify=lambda: (False, "FAIL: test_x (m.T.test_x)\nFAILED (failures=1)"), verify_rounds=3)
        a.run("việc")
        msgs = [m["content"] for m in a.messages if "[Kiểm tra tự động]" in (m.get("content") or "")]
        self.assertNotIn("GIỐNG HỆT", msgs[0])      # lần đầu chưa gợi ý
        self.assertIn("GIỐNG HỆT", msgs[1])         # lặp lại thì gợi ý xét chính test
        self.assertIn("ngày giờ cố định", msgs[1])


class HysteresisTests(unittest.TestCase):
    def test_compacts_once_then_stays_stable(self):
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup); ws = Workspace(d.name)
        a = Agent(SmartClient([]), build_tools(ws), ws, ui=UI(io.StringIO()), max_context_chars=4000)
        a.turn = 1
        a.messages.append({"role": "user", "content": "làm", "_turn": 1})
        for i in range(10):
            a.messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": str(i), "type": "function",
                               "function": {"name": "read_file", "arguments": "{}"}}]})
            a.messages.append({"role": "tool", "tool_call_id": str(i), "content": f"r{i} " + "z" * 450})
        self.assertGreater(a._size(), 4000)
        a.compact()
        self.assertLessEqual(a._size(), 2800)           # tụt xuống ~70% ngân sách, không chỉ vừa đủ dưới trần
        snapshot = json.dumps(a.messages, sort_keys=True)
        for _ in range(3):                              # các bước sau: không đổi gì => giữ được cache tiền tố của server
            a.messages.append({"role": "assistant", "content": "ok"})
            a.compact()
        self.assertTrue(json.dumps(a.messages[:-3], sort_keys=True) == snapshot)


class ExhaustedFlagTests(unittest.TestCase):
    def test_flag_distinguishes_budget_from_natural_end(self):
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup); ws = Workspace(d.name)
        loop = [Reply(tool_calls=[{"id": str(i), "name": "list_dir", "arguments": {"path": "."}}]) for i in range(3)]
        a = Agent(SmartClient(loop), build_tools(ws), ws, ui=UI(io.StringIO()), max_steps=3)
        a.run("x")
        self.assertTrue(a.exhausted)
        b = Agent(SmartClient([Reply(content="")] * 3), build_tools(ws), ws, ui=UI(io.StringIO()))
        b.run("x")                                       # kết thúc tự nhiên nhưng câu trả lời cuối RỖNG
        self.assertFalse(b.exhausted)


class GoalPlanOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory(); self.addCleanup(self.d.cleanup); self.ws = Workspace(self.d.name)

    def agent(self, script):
        return Agent(SmartClient(script), build_tools(self.ws), self.ws, ui=UI(io.StringIO()))

    def todo(self, text):
        return Reply(tool_calls=[{"id": "t", "name": "todo", "arguments": {"todos": text}}])

    def test_model_rewriting_todo_cannot_resurrect_or_shift_items(self):
        from harness.goal import run_goal
        loop = [Reply(tool_calls=[{"id": str(i), "name": "list_dir", "arguments": {"path": str(i)}}]) for i in range(3)]
        a = self.agent([self.todo("[ ] việc A\n[ ] việc A\n[ ] việc B"), Reply(content="kế hoạch"),   # có việc trùng
                        self.todo("[ ] việc A mới toanh\n[ ] việc Z"),                                   # model viết lại cả danh sách
                        *loop,                                                                           # việc A: hết bước
                        Reply(content="xong B")])
        res = run_goal(a, "mục tiêu", step_budget=4, plan_steps=5, require_change=False)
        self.assertEqual([(r.index, r.text, r.status) for r in res], [(1, "việc A", "blocked"), (2, "việc B", "done")])
        self.assertEqual(a.tools["todo"].state, ["[!] việc A (bị chặn: hết 4 bước)", "[x] việc B"])   # kế hoạch chuẩn

    def test_empty_final_answer_is_not_blocked(self):
        from harness.goal import run_goal
        a = self.agent([self.todo("[ ] việc A"), Reply(content="kế hoạch"), Reply(content=""), Reply(content=""), Reply(content="")])
        self.assertEqual([r.status for r in run_goal(a, "mục tiêu", step_budget=4, plan_steps=5, require_change=False)], ["done"])

    def test_planning_phase_is_read_only(self):
        from harness.goal import run_goal
        a = self.agent([Reply(tool_calls=[{"id": "w", "name": "write_file", "arguments": {"path": "x.py", "content": "1"}}]),
                        self.todo("[ ] việc A"), Reply(content="kế hoạch"), Reply(content="xong")])
        run_goal(a, "mục tiêu", step_budget=4, plan_steps=6, require_change=False)
        self.assertFalse((self.ws.root / "x.py").exists())          # không ghi được khi đang lập kế hoạch
        self.assertIn("write_file", a.tools)                        # công cụ đầy đủ được khôi phục sau đó


class GoalGuardsTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory(); self.addCleanup(self.d.cleanup); self.ws = Workspace(self.d.name)

    def agent(self, script, **kw):
        return Agent(SmartClient(script), build_tools(self.ws), self.ws, ui=UI(io.StringIO()), **kw)

    def todo(self, text):
        return Reply(tool_calls=[{"id": "t", "name": "todo", "arguments": {"todos": text}}])

    def write(self, path, content="x = 1\n"):
        return Reply(tool_calls=[{"id": "w" + path, "name": "write_file", "arguments": {"path": path, "content": content}}])

    def test_claiming_done_without_changing_files_is_not_done(self):
        from harness.goal import run_goal
        # báo xong -> bị yêu cầu bổ sung 2 lần -> vẫn không đổi gì -> bị chặn
        a = self.agent([self.todo("[ ] làm A"), Reply(content="kế hoạch"), Reply(content="đã xong hết"),
                        Reply(content="vẫn xong"), Reply(content="thật sự xong")])
        res = run_goal(a, "mục tiêu", step_budget=4, plan_steps=5)
        self.assertEqual([r.status for r in res], ["blocked"])
        self.assertIn("CHƯA thay đổi file nào", res[0].note)
        self.assertEqual(sum("CHƯA đạt" in (m.get("content") or "") for m in a.messages), 2)   # có đúng 2 lần bổ sung

    def test_code_without_test_gets_a_second_chance_then_passes(self):
        from harness.goal import run_goal
        a = self.agent([self.todo("[ ] viết module m\n[ ] viết module n"), Reply(content="kế hoạch"),
                        self.write("m.py"), Reply(content="xong m"),                        # chỉ đổi mã, không test -> bổ sung
                        self.write("test_m.py"), Reply(content="đã thêm test"),             # bổ sung đạt
                        self.write("n.py"), self.write("test_n.py"), Reply(content="xong n")])
        res = run_goal(a, "mục tiêu", step_budget=5, plan_steps=5)
        self.assertEqual([r.status for r in res], ["done", "done"])
        self.assertIn("test_m.py", res[0].note)
        self.assertTrue(any("chưa thêm/sửa test" in (m.get("content") or "") for m in a.messages))

    def test_code_without_test_blocked_when_never_fixed_and_flag_disables_rule(self):
        from harness.goal import run_goal
        script = lambda: [self.todo("[ ] viết module m"), Reply(content="kế hoạch"), self.write("m.py"), Reply(content="xong"),
                          Reply(content="xong rồi"), Reply(content="xong thật")]
        res = run_goal(self.agent(script()), "mục tiêu", step_budget=5, plan_steps=5)
        self.assertEqual(res[0].status, "blocked")
        self.assertIn("chưa thêm/sửa test", res[0].note)
        self.ws.root.joinpath("m.py").unlink()
        res2 = run_goal(self.agent(script()[:4]), "mục tiêu", step_budget=5, plan_steps=5, require_tests=False)
        self.assertEqual(res2[0].status, "done")

    def test_verify_off_during_planning_and_baseline_frozen(self):
        from harness.goal import run_goal
        calls = []
        outs = ["FAIL: test_cu (m.T.test_cu)\nFAILED (failures=1)"] + ["FAIL: test_cu (m.T.test_cu)\nFAIL: test_moi (m.T.test_moi)\nFAILED (failures=2)"] * 6
        def verify():
            calls.append(1)
            return False, outs[min(len(calls) - 1, len(outs) - 1)]
        a = self.agent([self.todo("[ ] việc A - test"), Reply(content="kế hoạch"), self.write("a.py"), self.write("test_a.py"),
                        Reply(content="xong A"), Reply(content="sửa rồi"), Reply(content="thôi")], verify=verify, verify_rounds=2)
        res = run_goal(a, "mục tiêu", step_budget=6, plan_steps=5)
        self.assertEqual([r.status for r in res], ["blocked"])
        self.assertEqual(len(calls), 5)   # 1 baseline + 1 sau việc A + 2 vòng sửa (kể cả vòng cuối) + 1 kiểm tra cuối; KHÔNG chạy khi lập kế hoạch
        self.assertTrue(any("[Kiểm tra tự động]" in (m.get("content") or "") and "test_moi" in m["content"] for m in a.messages))


class LedgerTests(unittest.TestCase):
    def test_evicted_actions_stay_visible_in_system_prompt(self):
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup); ws = Workspace(d.name)
        a = Agent(SmartClient([]), build_tools(ws), ws, ui=UI(io.StringIO()), max_context_chars=2500, summarize=True)
        a.turn = 1
        a.messages.append({"role": "user", "content": "làm", "_turn": 1})
        def call(i, name, args):
            a.messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": str(i), "type": "function",
                               "function": {"name": name, "arguments": json.dumps(args)}}]})
            a.messages.append({"role": "tool", "tool_call_id": str(i), "content": "ok " + "z" * 400})
        call(0, "bash", {"command": "cat >> tests/test_db.py << EOF class TestScheduler EOF"})
        call(1, "bash", {"command": "cat >> tests/test_db.py << EOF class TestScheduler EOF"})
        call(2, "write_file", {"path": "people_memory/memory.py", "content": "a\nb\n"})
        call(3, "read_file", {"path": "x"})                      # đọc thì không phải hành động có tác dụng phụ
        for i in range(4, 12):
            call(i, "read_file", {"path": "y" + str(i)})
        a.compact()
        sysmsg = a.messages[0]["content"]
        self.assertIn("NHẬT KÝ VIỆC BẠN ĐÃ LÀM", sysmsg)
        self.assertIn("tests/test_db.py", sysmsg)
        self.assertIn("(×2)", sysmsg)                             # lặp liên tiếp được gộp, để model thấy mình đang lặp
        self.assertIn("write_file: people_memory/memory.py (2 dòng)", sysmsg)
        self.assertNotIn("read_file", a.ledger[0])

    def test_ledger_persists_across_save_load(self):
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup); ws = Workspace(d.name)
        a = Agent(SmartClient([]), build_tools(ws), ws, ui=UI(io.StringIO()))
        a.ledger = ["bash: echo 1"]
        a.save(ws.root / "s.json")
        b = Agent(SmartClient([]), build_tools(ws), ws, ui=UI(io.StringIO()))
        b.load(ws.root / "s.json")
        self.assertEqual(b.ledger, ["bash: echo 1"])
        self.assertIn("echo 1", b.messages[0]["content"])


class RepetitionGuardTests(unittest.TestCase):
    def test_detect_repetition(self):
        from harness.client import detect_repetition
        loop = "Phân tích ban đầu. " + "Vòng lặp đệ quy! Mỗi lần gọi connect() lại gọi init_table(). " * 5
        self.assertTrue(detect_repetition(loop))
        self.assertFalse(detect_repetition("Câu một dài hơn hai mươi ký tự. Câu hai khác hẳn. Câu ba cũng khác. Câu bốn nữa."))
        self.assertFalse(detect_repetition("Ok. Ok. Ok. Ok. Ok."))   # câu ngắn lặp (vd danh sách) không bị coi là vòng lặp
        self.assertFalse(detect_repetition(""))

    def test_client_aborts_stream_early(self):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from harness.client import LMStudioClient
        sent = {"n": 0}

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_POST(self):
                self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
                try:
                    for _ in range(400):
                        chunk = {"choices": [{"delta": {"reasoning_content": "Vòng lặp đệ quy! Mỗi lần gọi connect() lại gọi init(). "}}]}
                        self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()); self.wfile.flush()
                        sent["n"] += 1
                    self.wfile.write(b"data: [DONE]\n\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        r = LMStudioClient("m", f"http://127.0.0.1:{srv.server_port}").chat([{"role": "user", "content": "x"}])
        self.assertEqual(r.aborted, "repetition")
        self.assertLess(len(r.thinking), 2000)     # ngắt sớm, không đọc hết 400 chunk

    def test_agent_nudges_after_aborted_reply_and_continues(self):
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup); ws = Workspace(d.name)
        stuck = Reply(content="", aborted="repetition")
        a = Agent(SmartClient([stuck, Reply(content="đã chọn một hành động và xong")]), build_tools(ws), ws, ui=UI(io.StringIO()))
        self.assertEqual(a.run("việc"), "đã chọn một hành động và xong")
        self.assertTrue(any("[Hệ thống]" in (m.get("content") or "") and "MỘT hành động" in m["content"] for m in a.messages))


class DocOnlyItemTests(unittest.TestCase):
    def test_readme_item_does_not_need_test_file(self):
        from harness.goal import run_goal
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup); ws = Workspace(d.name)
        todo = Reply(tool_calls=[{"id": "t", "name": "todo", "arguments": {"todos": "[ ] viết README ghi test nào chứng minh"}}])
        w = Reply(tool_calls=[{"id": "w", "name": "write_file", "arguments": {"path": "README.md", "content": "# x\n"}}])
        a = Agent(SmartClient([todo, Reply(content="kế hoạch"), w, Reply(content="xong")]), build_tools(ws), ws, ui=UI(io.StringIO()))
        self.assertEqual([r.status for r in run_goal(a, "mục tiêu", step_budget=4, plan_steps=5)], ["done"])


class EmptyReplyTests(unittest.TestCase):
    def test_empty_reply_is_retried_with_nudge(self):
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup); ws = Workspace(d.name)
        a = Agent(SmartClient([Reply(content=""), Reply(content="  "), Reply(content="đã làm xong")]), build_tools(ws), ws,
                  ui=UI(io.StringIO()))
        self.assertEqual(a.run("việc"), "đã làm xong")
        self.assertEqual(sum("[Hệ thống] Bạn chưa trả lời gì" in (m.get("content") or "") for m in a.messages), 2)

    def test_gives_up_after_two_retries(self):
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup); ws = Workspace(d.name)
        a = Agent(SmartClient([Reply(content="")] * 3), build_tools(ws), ws, ui=UI(io.StringIO()))
        self.assertEqual(a.run("việc"), "")     # không lặp vô hạn


class ClientHTTPLevelTests(unittest.TestCase):
    """Test client qua HTTP thật (không dùng client giả) để bắt lỗi trong chính lớp stream."""
    def serve(self, chunks):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_POST(self):
                self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
                for c in chunks:
                    self.wfile.write(f"data: {json.dumps(c, ensure_ascii=False)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        from harness.client import LMStudioClient
        return LMStudioClient("m", f"http://127.0.0.1:{srv.server_port}")

    def test_tool_calls_are_assembled_from_stream(self):
        client = self.serve([
            {"choices": [{"delta": {"reasoning_content": "nghĩ một chút"}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "read_file", "arguments": "{\"pa"}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "th\": \"a.py\"}"}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 1, "id": "c2", "function": {"name": "list_dir", "arguments": "{}"}}]}}]},
            {"choices": [], "usage": {"prompt_tokens": 123, "completion_tokens": 9}},
        ])
        r = client.chat([{"role": "user", "content": "x"}])
        self.assertEqual([(c["id"], c["name"], c["arguments"]) for c in r.tool_calls],
                         [("c1", "read_file", {"path": "a.py"}), ("c2", "list_dir", {})])
        self.assertEqual((r.thinking, r.prompt_tokens, r.aborted), ("nghĩ một chút", 123, ""))

    def test_plain_content_stream(self):
        client = self.serve([{"choices": [{"delta": {"content": "Xin "}}]}, {"choices": [{"delta": {"content": "chào"}}]}])
        self.assertEqual(client.chat([{"role": "user", "content": "x"}]).content, "Xin chào")


class ChangedFilesTests(unittest.TestCase):
    def test_deleting_junk_db_counts_but_creating_or_touching_does_not(self):
        from harness.goal import changed_files
        before = {"a.py": (1, 1), "x.db": (5, 1), "y.db": (5, 1)}
        after = {"a.py": (1, 1), "y.db": (9, 2), "z.db": (3, 3)}      # x.db bị xoá; y.db bị sửa; z.db mới sinh ra
        self.assertEqual(changed_files(before, after), ["x.db"])
        self.assertEqual(changed_files({"a.py": (1, 1)}, {"a.py": (2, 2), "z.db": (1, 1)}), ["a.py"])
        self.assertEqual(changed_files({"a.py": (1, 1)}, {"a.py": (1, 1), "z.db": (1, 1)}), [])


class RepairAfterExhaustionTests(unittest.TestCase):
    def test_exhausted_item_leaving_red_suite_triggers_repair_before_next_item(self):
        from harness.goal import run_goal
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup); ws = Workspace(d.name)
        outs = [(True, "OK"), (False, "FAIL: test_x (m.T.test_x)\nFAILED (failures=1)"), (True, "OK"), (True, "OK")]
        calls = []
        def verify():
            calls.append(1)
            return outs[min(len(calls) - 1, len(outs) - 1)]
        loop = [Reply(tool_calls=[{"id": str(i), "name": "list_dir", "arguments": {"path": str(i)}}]) for i in range(3)]
        todo = Reply(tool_calls=[{"id": "t", "name": "todo", "arguments": {"todos": "[ ] việc A"}}])
        a = Agent(SmartClient([todo, Reply(content="kế hoạch"), *loop, Reply(content="đã sửa test, xanh rồi")]),
                  build_tools(ws), ws, ui=UI(io.StringIO()), verify=verify)
        res = run_goal(a, "mục tiêu", step_budget=3, plan_steps=5, require_change=False)
        self.assertEqual(res[0].status, "blocked")                            # việc vẫn bị ghi là hết bước
        self.assertIn("đã yêu cầu sửa", res[0].note)
        self.assertTrue(any("bộ test ĐỎ" in (m.get("content") or "") and "test_x" in m["content"] for m in a.messages))


FAKE_ORACLE = '''import sys, json, pathlib
root = pathlib.Path(sys.argv[sys.argv.index("--root") + 1]) if "--root" in sys.argv else pathlib.Path(".")
if "--list" in sys.argv:
    print(json.dumps([{"id": 1, "title": "A", "brief": "tạo a.txt"}, {"id": 2, "title": "B", "brief": "tạo b.txt"}])); sys.exit(0)
upto = int(sys.argv[sys.argv.index("--upto") + 1])
bad = [n for n in ["a.txt", "b.txt"][:upto] if not (root / n).exists()]
for n in bad:
    print(f"✗ [S1] cần {n}\\n   vấn đề: file {n} chưa có")
print("✓ [S1] cái đã đạt")
print("KẾT QUẢ:", "CHƯA ĐẠT" if bad else "ĐẠT"); sys.exit(1 if bad else 0)
'''


class StagesTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory(); self.addCleanup(self.d.cleanup); self.ws = Workspace(self.d.name)
        self.script = self.ws.root.parent / (self.ws.root.name + "_oracle.py")
        self.script.write_text(FAKE_ORACLE)
        self.addCleanup(self.script.unlink)

    def agent(self, script):
        return Agent(SmartClient(script), build_tools(self.ws), self.ws, ui=UI(io.StringIO()))

    def w(self, path):
        return Reply(tool_calls=[{"id": "w" + path, "name": "write_file", "arguments": {"path": path, "content": "x\n"}}])

    def test_condense_keeps_failures_only(self):
        from harness.stages import condense
        out = "✓ [S1] tốt\n✗ [S2] hỏng\n   vấn đề: x\n   nhận được: y\n✓ [S3] tốt\nGIAI ĐOẠN 1: ĐẠT (1/1)\nKẾT QUẢ: CHƯA ĐẠT"
        self.assertEqual(condense(out), "✗ [S2] hỏng\n   vấn đề: x\n   nhận được: y\nGIAI ĐOẠN 1: ĐẠT (1/1)\nKẾT QUẢ: CHƯA ĐẠT")

    def test_runs_stages_in_order_and_gates_completion(self):
        from harness.stages import Oracle, run_stages
        a = self.agent([self.w("a.txt"), Reply(content="xong 1"),
                        self.w("b.txt"), Reply(content="xong 2")])
        res = run_stages(a, Oracle(self.script, self.ws.root), step_budget=5, max_rounds=2)
        self.assertEqual([(r.id, r.ok) for r in res], [(1, True), (2, True)])
        self.assertTrue(any("GIAI ĐOẠN 1/2" in (m.get("content") or "") and "file a.txt chưa có" in m["content"] for m in a.messages))  # có đưa lỗi cụ thể

    def test_completion_claim_is_rejected_until_oracle_passes(self):
        from harness.stages import Oracle, run_stages
        a = self.agent([Reply(content="xong rồi"),                 # nói xong nhưng chưa làm gì: cổng oracle trả lại
                        self.w("a.txt"), Reply(content="giờ xong"),
                        self.w("b.txt"), Reply(content="xong 2")])
        res = run_stages(a, Oracle(self.script, self.ws.root), step_budget=6, max_rounds=2)
        self.assertEqual([r.ok for r in res], [True, True])
        self.assertTrue(any("[Kiểm tra tự động]" in (m.get("content") or "") for m in a.messages))

    def test_stops_when_a_stage_never_passes(self):
        from harness.stages import Oracle, run_stages
        a = self.agent([Reply(content="xong")] * 12)
        res = run_stages(a, Oracle(self.script, self.ws.root), step_budget=4, max_rounds=1)
        self.assertEqual([(r.id, r.ok) for r in res], [(1, False)])    # không sang giai đoạn 2 khi giai đoạn 1 chưa đạt
        self.assertIn("✗ [S1] cần a.txt", res[0].output)


# ---------------------------------------------------------------- độ bền khi server model lỗi giữa chừng
class ClientFailureTests(unittest.TestCase):
    """Lỗi giữa chừng khi stream phải thành LLMError(transient), không được lọt ra thành TimeoutError/JSONDecodeError/..."""
    def client(self, mode):
        import socket
        import threading
        import time
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_POST(self):
                self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"xin "}}]}\n\n'); self.wfile.flush()
                try:
                    if mode == "stall":
                        time.sleep(2)
                    elif mode == "badjson":
                        self.wfile.write(b"data: {not json\n\n"); self.wfile.flush()
                    elif mode == "reset":
                        self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")
                        self.connection.close()
                except OSError:
                    pass
        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close); self.addCleanup(srv.shutdown)
        from harness.client import LMStudioClient
        return LMStudioClient("m", f"http://127.0.0.1:{srv.server_port}", timeout=0.5)

    def check(self, mode):
        from harness.client import LLMError
        with self.assertRaises(LLMError) as cm:
            self.client(mode).chat([{"role": "user", "content": "x"}])
        self.assertTrue(cm.exception.transient)

    def test_stalled_stream_times_out_as_llm_error(self): self.check("stall")
    def test_garbled_stream_line_is_llm_error(self): self.check("badjson")
    def test_connection_reset_mid_stream_is_llm_error(self): self.check("reset")

    def test_http_error_is_not_transient(self):
        from harness.client import LLMError, LMStudioClient
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_POST(self):
                self.send_response(400); self.end_headers(); self.wfile.write(b"bad request")
        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close); self.addCleanup(srv.shutdown)
        with self.assertRaises(LLMError) as cm:
            LMStudioClient("m", f"http://127.0.0.1:{srv.server_port}").chat([{"role": "user", "content": "x"}])
        self.assertFalse(cm.exception.transient)


class NetRetryTests(unittest.TestCase):
    class Dead:
        """Client luôn lỗi; đếm số lần bị gọi."""
        def __init__(self, transient): self.calls, self.transient = 0, transient
        def chat(self, messages, tools=None, on_token=None, on_thinking=None):
            from harness.client import LLMError
            self.calls += 1
            raise LLMError("đứt kết nối", transient=self.transient)

    def agent(self, client):
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup); ws = Workspace(d.name)
        a = Agent(client, build_tools(ws), ws, ui=UI(io.StringIO()))
        a.NET_RETRY_DELAY = 0
        return a

    def test_transient_error_is_retried_once_then_succeeds(self):
        from harness.client import LLMError
        c = SmartClient([Reply(content="xong")], fail_first=LLMError("đứt", transient=True))
        self.assertEqual(self.agent(c).run("chào"), "xong")

    def test_persistent_transient_error_is_raised_after_one_retry(self):
        from harness.client import LLMError
        c = self.Dead(transient=True)
        with self.assertRaises(LLMError):
            self.agent(c).run("chào")
        self.assertEqual(c.calls, 2)

    def test_non_transient_error_is_not_retried(self):
        from harness.client import LLMError
        c = self.Dead(transient=False)
        with self.assertRaises(LLMError):
            self.agent(c).run("chào")
        self.assertEqual(c.calls, 1)

    def test_failed_turn_leaves_no_dangling_user_message(self):
        from harness.client import LLMError
        a = self.agent(self.Dead(transient=False))
        with self.assertRaises(LLMError):
            a.run("chào")
        self.assertTrue(a.drop_dangling_turn())
        self.assertEqual((a.turn, [m["role"] for m in a.messages]), (0, ["system"]))
        self.assertFalse(a.drop_dangling_turn())   # không còn gì để bỏ

    def test_turn_that_already_did_work_is_kept(self):
        from harness.client import LLMError

        class Once(self.Dead):
            def chat(s, *a, **k):
                if s.calls == 0:
                    s.calls += 1
                    return Reply(tool_calls=[{"id": "1", "name": "list_dir", "arguments": {"path": "."}}])
                return super().chat(*a, **k)
        a = self.agent(Once(transient=False))
        with self.assertRaises(LLMError):
            a.run("liệt kê")
        self.assertFalse(a.drop_dangling_turn())   # đã có tool_call + kết quả: giữ nguyên, model tiếp tục từ đó


class GoalSavesProgressTests(unittest.TestCase):
    def test_on_item_done_called_after_each_item(self):
        from harness.goal import run_goal
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup); ws = Workspace(d.name)
        todo = lambda t: Reply(tool_calls=[{"id": "t", "name": "todo", "arguments": {"todos": t}}])
        a = Agent(SmartClient([todo("[ ] A\n[ ] B"), Reply(content="kế hoạch"), Reply(content="xong A"), Reply(content="xong B")]),
                  build_tools(ws), ws, ui=UI(io.StringIO()))
        seen = []
        run_goal(a, "mục tiêu", step_budget=3, require_change=False, on_item_done=lambda: seen.append(a.turn))
        self.assertEqual(seen, [2, 3])   # gọi sau việc 1 (lượt 2) và sau việc 2 (lượt 3)

    def test_cli_goal_keeps_session_when_server_dies_mid_run(self):
        """Chạy CLI thật với server giả: việc 1 xong, rồi server ngắt kết nối. Phiên phải còn trên đĩa và lệnh trả mã 1."""
        import contextlib
        import socket
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from unittest import mock

        calls = []

        def sse(delta):
            return ("data: " + json.dumps({"choices": [{"delta": delta}]}, ensure_ascii=False) + "\n\n").encode()

        def tool(name, args):
            return sse({"tool_calls": [{"index": 0, "id": f"c{len(calls)}", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]})

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_GET(self):
                if self.path == "/v1/models":
                    body = json.dumps({"data": [{"id": "fake"}]}).encode()
                    self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)
                else:
                    self.send_response(404); self.end_headers()
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                calls.append(1)
                script = {1: tool("todo", {"todos": "[ ] việc A\n[ ] việc B"}), 2: sse({"content": "kế hoạch xong"}),
                          3: tool("write_file", {"path": "a.txt", "content": "x\n"}), 4: sse({"content": "xong việc A"})}
                if len(calls) not in script:   # từ yêu cầu thứ 5: server "chết" giữa chừng
                    self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")
                    self.connection.close()
                    return
                self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
                self.wfile.write(script[len(calls)]); self.wfile.write(b"data: [DONE]\n\n")
        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close); self.addCleanup(srv.shutdown)
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup)

        from harness import cli
        out = io.StringIO()
        with mock.patch.object(cli, "load_mcp_tools", lambda *a, **k: ({}, [])), \
                mock.patch.object(Agent, "NET_RETRY_DELAY", 0), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            rc = cli.main(["--host", f"http://127.0.0.1:{srv.server_port}", "-m", "fake", "-C", d.name, "--no-memory",
                           "--no-embed", "--no-summary", "--goal", "mục tiêu", "--step-budget", "4", "--max-items", "2"])
        self.assertEqual(rc, 1)
        saved = Path(d.name, ".harness", "sessions", "last.json")
        self.assertTrue(saved.exists(), out.getvalue())
        self.assertIn("xong việc A", saved.read_text())
        self.assertGreaterEqual(len(calls), 6)   # yêu cầu thứ 5 bị ngắt, đã thử lại một lần (yêu cầu thứ 6)


class StageExitCodeTests(unittest.TestCase):
    def test_exit_code_uses_real_last_stage_not_a_magic_number(self):
        from harness.stages import StageResult, exit_code
        ok = lambda i: StageResult(i, "t", True, 0, "")
        self.assertEqual(exit_code([ok(1), ok(2)], 2), 0)                      # oracle chỉ có 2 giai đoạn: đạt cả hai là xong
        self.assertEqual(exit_code([ok(1), ok(2)], 7), 2)                      # chưa tới giai đoạn cuối
        self.assertEqual(exit_code([ok(1), StageResult(2, "t", False, 4, "")], 2), 2)
        self.assertEqual(exit_code([], 2), 2)
        self.assertEqual(exit_code([ok(1)], 0), 2)                             # oracle không liệt kê được giai đoạn nào


# ---------------------------------------------------------------- bí mật và quyền duyệt
class EnvScrubTests(unittest.TestCase):
    def test_clean_env_drops_secret_looking_names_only(self):
        from harness.sandbox import clean_env
        env = {"PATH": "/bin", "HOME": "/h", "LANG": "C", "GIT_AUTHOR_NAME": "bạn", "PYTHONPATH": "/x",
               "GITHUB_TOKEN": "1", "OPENAI_API_KEY": "1", "AWS_SECRET_ACCESS_KEY": "1", "AWS_ACCESS_KEY_ID": "1",
               "DB_PASSWORD": "1", "SSH_AUTH_SOCK": "1", "MY_SERVICE_KEY": "1", "SESSION_COOKIE": "1"}
        self.assertEqual(sorted(clean_env(env)), ["GIT_AUTHOR_NAME", "HOME", "LANG", "PATH", "PYTHONPATH"])

    def test_allow_list_keeps_named_variables(self):
        from harness.sandbox import clean_env
        env = {"OPENAI_API_KEY": "1", "DB_PASSWORD": "1", "GITHUB_TOKEN": "1", "HARNESS_ENV_ALLOW": "OPENAI_API_KEY, DB_PASSWORD"}
        self.assertEqual(sorted(k for k in clean_env(env) if k != "HARNESS_ENV_ALLOW"), ["DB_PASSWORD", "OPENAI_API_KEY"])

    @unittest.skipUnless(__import__("harness.sandbox", fromlist=["x"]).available(), "kernel không có Landlock")
    def test_bash_does_not_see_secret_env_unless_unsandboxed(self):
        import os
        from unittest import mock
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup); ws = Workspace(d.name)
        show = 'echo "[${FAKE_API_KEY:-vắng}]"'
        with mock.patch.dict(os.environ, {"FAKE_API_KEY": "s3cr3t-value"}):
            self.assertIn("[vắng]", build_tools(ws)["bash"].fn(show))
            self.assertIn("[s3cr3t-value]", build_tools(ws, sandbox=False)["bash"].fn(show))
            with mock.patch.dict(os.environ, {"HARNESS_ENV_ALLOW": "FAKE_API_KEY"}):
                self.assertIn("[s3cr3t-value]", build_tools(ws)["bash"].fn(show))


class ApprovalTests(unittest.TestCase):
    def test_web_tools_need_approval(self):
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup)
        tools = build_tools(Workspace(d.name))
        for name in ("web_search", "web_fetch", "bash", "write_file"):
            self.assertTrue(tools[name].needs_approval, name)
        for name in ("read_file", "grep", "todo"):
            self.assertFalse(tools[name].needs_approval, name)

    def test_subagent_inherits_ask_mode(self):
        """--ask: subagent do `delegate` tạo ra cũng phải hỏi trước khi chạm web, không được lách quyền duyệt."""
        from harness.agent import delegate_tool
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup); ws = Workspace(d.name)
        asked = []
        client = FakeClient([Reply(tool_calls=[{"id": "1", "name": "web_fetch", "arguments": {"url": "http://example.invalid/"}}]),
                             Reply(content="bị từ chối")])
        parent = Agent(client, build_tools(ws), ws, ui=UI(io.StringIO()), auto_approve=False,
                       approve=lambda name, args: asked.append(name) or False)
        self.assertEqual(delegate_tool(parent).fn(task="đọc trang web"), "bị từ chối")
        self.assertEqual(asked, ["web_fetch"])


class ReplSurvivesServerErrorTest(unittest.TestCase):
    def test_repl_reports_error_and_keeps_going(self):
        """Server ngắt kết nối giữa lượt: REPL báo lỗi, bỏ tin nhắn treo, vẫn nhận câu kế và thoát bằng /exit (mã 0)."""
        import contextlib
        import socket
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from unittest import mock

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_GET(self):
                if self.path == "/v1/models":
                    body = json.dumps({"data": [{"id": "fake"}]}).encode()
                    self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)
                else:
                    self.send_response(404); self.end_headers()
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")
                self.connection.close()
        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close); self.addCleanup(srv.shutdown)
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup)

        from harness import cli
        out = io.StringIO()
        with mock.patch.object(cli, "load_mcp_tools", lambda *a, **k: ({}, [])), \
                mock.patch.object(Agent, "NET_RETRY_DELAY", 0), \
                mock.patch("builtins.input", side_effect=["chào", "xin chào lần hai", "/exit"]) as inp, \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            rc = cli.main(["--host", f"http://127.0.0.1:{srv.server_port}", "-m", "fake", "-C", d.name, "--no-memory", "--no-embed"])
        self.assertEqual(rc, 0, out.getvalue())
        self.assertEqual(inp.call_count, 3)   # sau lỗi lượt 1, vẫn hỏi tiếp lượt 2 rồi mới /exit
        self.assertEqual(out.getvalue().count("gõ lại"), 2)


# ---------------------------------------------------------------- các test hồi quy bổ sung
class SecurityBoundaryAndToolsTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory(); self.addCleanup(self.d.cleanup)
        self.outside = tempfile.TemporaryDirectory(); self.addCleanup(self.outside.cleanup)
        self.ws = Workspace(self.d.name)
        self.tools = build_tools(self.ws)

    def test_harness_dir_and_symlinks_blocked(self):
        (self.ws.root / ".harness").mkdir()
        (self.ws.root / ".harness" / "mcp.json").write_text('{"mcpServers":{"x":{"command":"id"}}}')
        (self.ws.root / "ok.txt").write_text("noi dung thuong\n")
        (Path(self.outside.name) / "secret.txt").write_text("SECRET_OUTSIDE\n")
        (self.ws.root / "esc_file").symlink_to(Path(self.outside.name) / "secret.txt")
        (self.ws.root / "esc_dir").symlink_to(Path(self.outside.name))

        for path in (".harness", ".harness/mcp.json", "esc_file", "esc_dir/secret.txt"):
            with self.assertRaises(ValueError):
                self.ws.path(path)
        for fn in (
            lambda: self.tools["read_file"].fn(".harness/mcp.json"),
            lambda: self.tools["read_file"].fn("esc_file"),
            lambda: self.tools["write_file"].fn(".harness/mcp.json", "{}"),
            lambda: self.tools["edit_file"].fn("esc_file", "x", "y"),
            lambda: self.tools["move_file"].fn("ok.txt", ".harness/x"),
            lambda: self.tools["delete_file"].fn("esc_file"),
        ):
            with self.assertRaises(ValueError):
                fn()
        self.assertIn("LỖI", self.tools["move_file"].fn(".", "moved_root"))
        a = Agent(FakeClient([]), self.tools, self.ws, ui=UI(io.StringIO()))
        self.assertIn("LỖI", a._execute("read_file", {"path": ".harness/mcp.json"}))

        listing = self.tools["list_dir"].fn(".")
        self.assertNotIn("mcp.json", listing)
        self.assertNotIn("esc_file", listing)
        self.assertNotIn("esc_dir", listing)
        self.assertIn("ok.txt", listing)
        (self.ws.root / "sub").mkdir()
        (self.ws.root / "sub" / "nested.txt").write_text("ben trong\n")
        glob_top = self.tools["glob"].fn("*")
        glob_deep = self.tools["glob"].fn("**/*.txt")
        self.assertIn("ok.txt", glob_top)
        self.assertIn("nested.txt", glob_deep)
        self.assertNotIn("mcp.json", glob_top + self.tools["glob"].fn("*.json"))
        self.assertNotIn("secret.txt", glob_top + glob_deep)
        self.assertNotIn("SECRET_OUTSIDE", self.tools["grep"].fn("SECRET_OUTSIDE"))
        self.assertNotIn("mcpServers", self.tools["grep"].fn("mcpServers"))

    def test_nullable_tool_args(self):
        from harness.tools import P, Tool
        t = Tool("t", "d", {"req_null": P("integer", "a", nullable=True),
                            "req": P("integer", "b"),
                            "opt": P("integer", "c", optional=True)}, lambda **k: "")
        self.assertEqual(t.clean_args({"req_null": None, "req": 1, "opt": None}), {"req_null": None, "req": 1})
        self.assertIn("LỖI: tham số req", t.clean_args({"req_null": 1, "req": None}))


class StageFixTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory(); self.addCleanup(self.d.cleanup)
        self.ws_dir = tempfile.TemporaryDirectory(); self.addCleanup(self.ws_dir.cleanup)
        self.ws = Workspace(self.ws_dir.name)

    def test_oracle_relative_script_with_different_cwd_and_errors(self):
        import os
        from harness.stages import Oracle, OracleError
        script = Path(self.d.name) / "my_oracle.py"
        script.write_text(FAKE_ORACLE)
        prev = os.getcwd()
        os.chdir(self.d.name)
        try:
            o = Oracle("my_oracle.py", self.ws.root)
            stages = o.list_stages()
        finally:
            os.chdir(prev)
        self.assertEqual([s["id"] for s in stages], [1, 2])
        self.assertTrue(Path(o.script).is_absolute())

        with self.assertRaises(OracleError):
            Oracle(Path(self.d.name) / "missing.py", self.ws.root).list_stages()
        bad = Path(self.d.name) / "bad.py"
        bad.write_text("print('not json')")
        with self.assertRaises(OracleError):
            Oracle(bad, self.ws.root).list_stages()

    def test_run_stages_does_not_double_run_gate(self):
        from harness.stages import Oracle, run_stages
        log = Path(self.d.name) / "calls.txt"
        script = Path(self.d.name) / "count_oracle.py"
        script.write_text(
            f"import sys, json, pathlib\n"
            f"if '--list' in sys.argv:\n"
            f"    print(json.dumps([{{'id': 1, 'title': 'A', 'brief': 'làm'}}])); sys.exit(0)\n"
            f"with open({str(log)!r}, 'a') as f: f.write('run\\n')\n"
            f"print('ĐẠT'); sys.exit(0)\n"
        )
        a = Agent(SmartClient([Reply(content="xong")]), build_tools(self.ws), self.ws, ui=UI(io.StringIO()))
        res = run_stages(a, Oracle(script, self.ws.root), step_budget=3, max_rounds=2)
        self.assertEqual([r.ok for r in res], [True])
        self.assertEqual(len(log.read_text().splitlines()), 1)   # chỉ chạy 1 lần trong gate() của agent.run, không chạy dư nữa


class AgentEdgeFixTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory(); self.addCleanup(self.d.cleanup)
        self.ws = Workspace(self.d.name)

    def test_verify_final_fix_round_is_checked(self):
        calls = []
        results = [(False, "FAIL: test_a"), (False, "FAIL: test_a"), (True, "OK")]
        def verify():
            calls.append(1)
            return results.pop(0)
        a = Agent(SmartClient([Reply(content="xong"), Reply(content="sửa 1"), Reply(content="sửa 2")]),
                  build_tools(self.ws), self.ws, ui=UI(io.StringIO()), verify=verify, verify_rounds=2)
        ans = a.run("việc")
        self.assertEqual(ans, "sửa 2")
        self.assertEqual(len(calls), 3)
        self.assertTrue(a.last_verify_ok)

    def test_text_tools_false_ignores_xml_tool_call_and_raw_decode(self):
        xml = 'ok <tool_call>{"name":"list_dir","arguments":{"path":"."}}</tool_call>'
        a = Agent(SmartClient([Reply(content=xml)]), build_tools(self.ws), self.ws, ui=UI(io.StringIO()), text_tools=False)
        self.assertEqual(a.run("x"), xml)
        self.assertFalse(any(m.get("role") == "tool" for m in a.messages))

        tricky = '<tool_call>{"name":"write_file","arguments":{"path":"a.txt","content":"</tool_call> bên trong"}}</tool_call>'
        parsed = parse_text_tool_calls(tricky)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["arguments"]["content"], "</tool_call> bên trong")
        self.assertEqual(parse_text_tool_calls('<tool_call>{"name":123,"arguments":{}}</tool_call>'), [])
        self.assertEqual(parse_text_tool_calls('<tool_call>{"name":"x","arguments":[1,2]}</tool_call>'), [])

    def test_compact_works_without_memory_or_summary_and_caps_system(self):
        a = Agent(SmartClient([]), build_tools(self.ws), self.ws, ui=UI(io.StringIO()),
                  memory=None, summarize=False, max_context_chars=3600)
        a.turn = 1
        a.messages.append({"role": "user", "content": "làm " + "u" * 300, "_turn": 1})
        for i in range(14):
            a.messages.append({"role": "assistant", "content": f"bước {i} " + "a" * 120,
                               "tool_calls": [{"id": str(i), "type": "function",
                                               "function": {"name": "write_file", "arguments": json.dumps({"path": f"f{i}.py", "content": "a\n" * 10})}}]})
            a.messages.append({"role": "tool", "tool_call_id": str(i), "content": f"res {i} " + ("z" * 400 if i < 11 else "z" * 60)})
        self.assertGreater(a._size(), 3600)
        a.compact()
        self.assertLessEqual(a._size(), 3600)
        self.assertLess(len(a.messages[0]["content"]), 3600)
        self.assertIn("write_file: f0.py", "\n".join(a.ledger))

        # _ledger_add ghi cả --text-tools và không sập khi arguments không phải dict
        b = Agent(SmartClient([]), build_tools(self.ws), self.ws, ui=UI(io.StringIO()), text_tools=True, max_context_chars=6000)
        b._ledger_add([{"role": "assistant", "content": '<tool_call>{"name":"write_file","arguments":{"path":"t.py","content":"a"}}</tool_call>'},
                       {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function", "function": {"name": "bash", "arguments": "not-json"}}]}])
        self.assertIn("write_file: t.py", "\n".join(b.ledger))
        c = Agent(SmartClient([]), build_tools(self.ws), self.ws, ui=UI(io.StringIO()), text_tools=False, max_context_chars=3000)
        c.ledger = [f"write_file: item_{i}.py ({'x' * 120})" for i in range(50)]
        c._apply_system()
        self.assertLess(len(c.messages[0]["content"]), c._budget())

    def test_empties_counter_resets_and_nudge_edit_after_todo(self):
        # 2 rỗng -> trả lời thật -> lại 2 rỗng -> vẫn hồi phục được (empties được reset)
        a = Agent(SmartClient([Reply(content=""), Reply(content=""), Reply(content="tốt"),
                               Reply(content=""), Reply(content=""), Reply(content="tốt lần 2")]),
                  build_tools(self.ws), self.ws, ui=UI(io.StringIO()))
        self.assertEqual(a.run("1"), "tốt")
        self.assertEqual(a.run("2"), "tốt lần 2")

        st = {"nudged": 0, "todo": 0, "edit": 0}
        self.assertIn("todo", a._nudge([("read_file", {})] * 20, 20, st))
        self.assertIn("chưa sửa được file nào", a._nudge([("read_file", {})] * 35, 35, st))

    def test_load_rejects_bad_types_and_fingerprint_pytest(self):
        from harness.agent import fingerprint
        bad_file = self.ws.root / "bad.json"
        bad_file.write_text('["not", "dict"]')
        a = Agent(SmartClient([]), build_tools(self.ws), self.ws, ui=UI(io.StringIO()))
        self.assertFalse(a.load(bad_file))
        bad_file.write_text('{"messages": [], "turn": "bad", "first_live_turn": 1}')
        self.assertFalse(a.load(bad_file))

        pt_out = "FAILED tests/test_mod.py::TestA::test_one - AssertionError\nERROR tests/test_mod.py::TestA::test_two"
        self.assertEqual(fingerprint(pt_out), frozenset({"tests/test_mod.py::TestA::test_one", "tests/test_mod.py::TestA::test_two"}))


class GoalAndCliFixTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory(); self.addCleanup(self.d.cleanup)
        self.ws = Workspace(self.d.name)

    def test_goal_verify_failure_blocks_restores_verify_and_test_detection(self):
        from harness.goal import _is_test_file, _model_marked_done, Item, completion_problem, run_goal
        self.assertTrue(_is_test_file("tests/helpers.py"))
        self.assertTrue(_is_test_file("pkg/foo_test.py"))
        self.assertFalse(_is_test_file("attestation.py"))
        self.assertFalse(_is_test_file("contest.py"))
        self.assertIn("chưa thêm/sửa test", completion_problem(["contest.py"], True, True))
        self.assertFalse(_model_marked_done(["[x] việc thật"], Item("---")))
        self.assertFalse(_model_marked_done(["[x] ---"], Item("việc thật")))

        todo = lambda t: Reply(tool_calls=[{"id": "t", "name": "todo", "arguments": {"todos": t}}])
        w = lambda p: Reply(tool_calls=[{"id": p, "name": "write_file", "arguments": {"path": p, "content": "x = 1\n"}}])
        # Baseline xanh -> việc A làm đỏ -> bị chặn + final_ok False
        outs1 = [(True, "OK"), (False, "FAIL: test_new (m.T.test_new)"), (False, "FAIL: test_new (m.T.test_new)"), (False, "FAIL: test_new (m.T.test_new)")]
        verify1 = lambda: outs1.pop(0) if outs1 else (False, "FAIL: test_new (m.T.test_new)")
        a = Agent(SmartClient([todo("[ ] làm A"), Reply(content="kế hoạch"), w("a.py"), w("test_a.py"),
                               Reply(content="xong"), Reply(content="vẫn lỗi")]),
                  build_tools(self.ws), self.ws, ui=UI(io.StringIO()), verify=verify1, verify_rounds=1)
        res = run_goal(a, "g", step_budget=5, plan_steps=5)
        self.assertEqual([r.status for r in res], ["blocked"])
        self.assertFalse(res.final_ok)

        # Kiểm tra finally khôi phục agent.verify nếu lỗi giữa chừng khi lập kế hoạch
        from harness.client import LLMError
        verify_stub = lambda: (False, "FAIL: test_fail (m.T.test_fail)")
        b = Agent(SmartClient([], fail_first=LLMError("HTTP 500")), build_tools(self.ws), self.ws,
                  ui=UI(io.StringIO()), verify=verify_stub)
        with self.assertRaises(LLMError):
            run_goal(b, "g", step_budget=4, plan_steps=4)
        self.assertIs(b.verify, verify_stub)

    def test_goal_resume_restores_plan_and_progress(self):
        from harness.goal import run_goal
        todo = lambda t: Reply(tool_calls=[{"id": "t", "name": "todo", "arguments": {"todos": t}}])
        a = Agent(SmartClient([todo("[ ] việc 1\n[ ] việc 2"), Reply(content="kế hoạch"),
                               Reply(content="xong 1")]),
                  build_tools(self.ws), self.ws, ui=UI(io.StringIO()))
        # chạy 1 việc rồi lưu bằng on_item_done khi việc 2 chưa có reply -> giả lập dừng sau việc 1
        saved = self.ws.root / "goal_sess.json"
        def stop_after_1():
            if len([1 for it in a.goal_state["items"] if it["status"] != "pending"]) == 1:
                a.save(saved)
                raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            run_goal(a, "g", step_budget=4, plan_steps=4, require_change=False, on_item_done=stop_after_1)

        b = Agent(SmartClient([Reply(content="xong 2")]), build_tools(self.ws), self.ws, ui=UI(io.StringIO()))
        self.assertTrue(b.load(saved))
        res = run_goal(b, "g", step_budget=4, plan_steps=4, require_change=False)
        self.assertEqual([(r.index, r.status) for r in res], [(1, "done"), (2, "done")])

    def test_cli_exit_codes_prechecks_eval_and_clear(self):
        import contextlib, subprocess, threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from unittest import mock
        from harness import cli

        # __main__ exit code != 0 + eval subcommand after flags + --help shows eval
        r_bad = subprocess.run([sys.executable, "-m", "harness", "--unknown-flag-xyz"], capture_output=True, text=True)
        self.assertNotEqual(r_bad.returncode, 0)
        r_help = subprocess.run([sys.executable, "-m", "harness", "--help"], capture_output=True, text=True)
        self.assertEqual(r_help.returncode, 0)
        self.assertIn("eval", r_help.stdout)
        r_eval = subprocess.run([sys.executable, "-m", "harness", "--host", "http://127.0.0.1:9", "eval", "--list"],
                                capture_output=True, text=True)
        self.assertEqual(r_eval.returncode, 0)
        self.assertIn("sandbox_escape", r_eval.stdout)

        # --resume <missing>, --system @missing, --oracle bad fail BEFORE backends()
        sess_dir = self.ws.root / ".harness" / "sessions"
        sess_dir.mkdir(parents=True)
        (sess_dir / "last.json").write_text('{"keep":"yes"}')
        called = []
        @contextlib.contextmanager
        def fail_if_started(a):
            called.append(1)
            yield a
        err_io = io.StringIO()
        with mock.patch.object(cli, "backends", fail_if_started), contextlib.redirect_stderr(err_io):
            self.assertEqual(cli.main(["-C", self.d.name, "--resume", "khong_ton_tai"]), 1)
            self.assertEqual(cli.main(["-C", self.d.name, "--system", "@/khong/co/file/nay.txt"]), 1)
            self.assertEqual(cli.main(["-C", self.d.name, "--oracle", "x.py"]), 1)
        self.assertEqual(called, [])
        self.assertEqual((sess_dir / "last.json").read_text(), '{"keep":"yes"}')

        # Server giả kiểm tra -p trả mã 2 khi verify fail, /clear lưu phiên, --threads với --local
        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_GET(self):
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
                self.wfile.write(json.dumps({"data": [{"id": "fake"}]}).encode())
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"xong"}}]}\n\ndata: [DONE]\n\n')
        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close); self.addCleanup(srv.shutdown)
        d2 = tempfile.TemporaryDirectory(); self.addCleanup(d2.cleanup)
        out = io.StringIO()
        with mock.patch.object(cli, "load_mcp_tools", lambda *a, **k: ({}, [])), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            rc = cli.main(["--host", f"http://127.0.0.1:{srv.server_port}", "-m", "fake", "-C", d2.name,
                           "--no-memory", "--no-embed", "--no-sandbox", "--verify", "exit 1", "--verify-rounds", "0", "-p", "hi"])
        self.assertEqual(rc, 2)
        self.assertTrue((Path(d2.name) / ".harness" / ".gitignore").exists())
        with mock.patch.object(cli, "load_mcp_tools", lambda *a, **k: ({}, [])), \
                mock.patch("builtins.input", side_effect=["chào", "/clear", "/exit"]), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            cli.main(["--host", f"http://127.0.0.1:{srv.server_port}", "-m", "fake", "-C", d2.name, "--no-memory", "--no-embed"])
        data = json.loads((Path(d2.name) / ".harness" / "sessions" / "last.json").read_text())
        self.assertEqual(len(data["messages"]), 1)

        # KeyboardInterrupt giữa REPL dọn tool_calls treo và lưu lại
        def interrupt_run(prompt):
            raise KeyboardInterrupt()
        with mock.patch.object(cli, "load_mcp_tools", lambda *a, **k: ({}, [])), \
                mock.patch.object(Agent, "run", lambda self, p: (
                    self.messages.extend([{"role": "user", "content": p, "_turn": self.turn + 1},
                                          {"role": "assistant", "content": "", "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "list_dir", "arguments": "{}"}}]}]),
                    setattr(self, "turn", self.turn + 1),
                    interrupt_run(p),
                )), \
                mock.patch("builtins.input", side_effect=["thử ngắt", "/exit"]), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            rc_int = cli.main(["--host", f"http://127.0.0.1:{srv.server_port}", "-m", "fake", "-C", d2.name, "--no-memory", "--no-embed"])
        self.assertEqual(rc_int, 0)
        d2_saved = json.loads((Path(d2.name) / ".harness" / "sessions" / "last.json").read_text())
        for idx, m in enumerate(d2_saved["messages"]):
            if m.get("role") == "assistant" and m.get("tool_calls"):
                self.assertEqual(d2_saved["messages"][idx + 1]["role"], "tool")

        # --local --threads 4 không bị nuốt khi không có --cpu
        spawned_threads = []
        class FakeLocal:
            def __init__(self, model, port, ctx, **kw):
                self.model, self.ctx, self.host = Path("m.gguf"), ctx, f"http://127.0.0.1:{srv.server_port}"
                spawned_threads.append(kw.get("threads"))
            def start(self): pass
            def stop(self): pass
        with mock.patch.object(cli, "find_model", lambda *a: Path("m.gguf")), \
                mock.patch.object(cli, "LocalServer", FakeLocal), \
                mock.patch.object(cli, "load_mcp_tools", lambda *a, **k: ({}, [])), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            cli.main(["--local", "m", "--threads", "4", "-C", d2.name, "--no-memory", "--no-embed", "-p", "ok"])
        self.assertEqual(spawned_threads, [4])


class MemoryAndComponentFixTests(unittest.TestCase):
    def test_memory_d_folding_migration_prelimit_vectors_and_backfill_error(self):
        import array, sqlite3
        from harness.embed import EmbedError
        from harness.memory import Memory, lexically_strong

        # đ/Đ folding + uppercase lexically_strong + STOP_FOLDED
        with Memory(":memory:", "s") as m:
            m.add("Điểm danh ở Đà Nẵng", "note", 1)
            self.assertEqual(len(m.search("diem danh da nang")), 1)
            self.assertEqual(len(m.search("ĐIỂM DANH")), 1)
            self.assertTrue(lexically_strong("Đọc file này đi", "DOC FILE NAY"))
            self.assertFalse(lexically_strong("Khong co gi", "KHONG NAY NAY"))

            # Pre-LIMIT filtering cho kinds và first_live_turn
            for i in range(40):
                m.add(f"python turn mới {i}", "turn", 10 + i)
            m.add("python note cũ quan trọng", "note", 1)
            hits = m.search("python", k=2, first_live_turn=10, kinds=("note",))
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0].kind, "note")

            # >5000 vectors: vector cũ nhất rid=1 vẫn tìm được
            class OneVec:
                def embed(self, texts):
                    return [array.array("f", [1.0, 0.0]) for _ in texts]
            m.embedder = OneVec()
            old_rid = m.db.execute("SELECT rowid FROM mem WHERE kind='note' LIMIT 1").fetchone()[0]
            m.db.execute("INSERT OR REPLACE INTO vec VALUES (?,?)", (old_rid, array.array("f", [1.0, 0.0]).tobytes()))
            other_blob = array.array("f", [0.0, 1.0]).tobytes()
            rows_bulk = []
            for j in range(5010):
                cur = m.db.execute("INSERT INTO mem(content, folded, kind, session, turn, ts) VALUES (?,?,?,?,?,?)",
                                   (f"junk {j}", f"junk {j}", "turn", "other", 1, 1.0))
                rows_bulk.append((cur.lastrowid, other_blob))
            m.db.executemany("INSERT INTO vec VALUES (?,?)", rows_bulk)
            m.db.commit()
            vhits = m.search("zzz_no_bm25_match", k=1, kinds=("note",))
            self.assertEqual(len(vhits), 1)

            # backfill() trả về 0 khi EmbedError
            class FailEmbed:
                def embed(self, texts): raise EmbedError("down")
            m.add("chưa nhúng", "note", 2)
            m.embedder = FailEmbed()
            self.assertEqual(m.backfill(), 0)

        # Di trú bảng 5 cột cũ giữ nguyên rowid
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup)
        dbp = Path(d.name) / "old.db"
        con = sqlite3.connect(dbp)
        con.execute('CREATE VIRTUAL TABLE mem USING fts5(content, kind UNINDEXED, session UNINDEXED, turn UNINDEXED, ts UNINDEXED)')
        con.execute("INSERT INTO mem(rowid, content, kind, session, turn, ts) VALUES (42, 'Điểm cũ', 'note', 's0', 1, 1.0)")
        con.commit(); con.close()
        with Memory(dbp, "s0") as m2:
            self.assertEqual(len(m2.search("diem cu")), 1)
            self.assertEqual(m2.db.execute("SELECT rowid FROM mem").fetchone()[0], 42)

    def test_client_invalid_args_null_chunks_embed_filter_ui_and_mcp(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import threading
        from harness.client import LLMError, LMStudioClient
        from harness.mcp import load_mcp_tools
        from harness.ui import UI

        # UI non-dict defensive
        ui = UI(io.StringIO())
        ui.tool_call("t", ["bad"])
        ui.ask_approval("t", ["bad"])

        # Client stream với bad JSON, non-dict args, delta=null, models filter
        chunks = [
            {"choices": [{"delta": None}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "read_file", "arguments": "{bad"}}]}}] },
            {"choices": [{"delta": {"tool_calls": [{"index": 1, "id": "c2", "function": {"name": "list_dir", "arguments": "[1,2]"}}]}}] },
        ]
        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_GET(self):
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
                self.wfile.write(json.dumps({"data": [{"id": "BAAI/bge-m3"}, {"id": "Qwen-Coder"}]}).encode())
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
                for c in chunks:
                    self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close); self.addCleanup(srv.shutdown)
        c = LMStudioClient(host=f"http://127.0.0.1:{srv.server_port}")
        self.assertEqual(c.ensure_model(), "Qwen-Coder")
        rep = c.chat([{"role": "user", "content": "x"}])
        self.assertIn("arg_error", rep.tool_calls[0])
        self.assertIn("arg_error", rep.tool_calls[1])
        self.assertEqual(rep.tool_calls[0]["arguments"], {})

        # MCP missing command + nullable schema
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup)
        (Path(d.name) / ".harness").mkdir()
        (Path(d.name) / ".harness" / "mcp.json").write_text(json.dumps(
            {"mcpServers": {"bad": {"args": []}, "ok": {"command": sys.executable, "args": [str(Path(__file__).parent / "fake_mcp.py")]}}}
        ))
        warns = []
        tools, servers = load_mcp_tools(Path(d.name), warn=warns.append)
        try:
            self.assertEqual(len(warns), 1)
            from harness.mcp import _wrap
            wt = _wrap(servers[0], {"name": "n", "inputSchema": {"properties": {"k": {"type": ["integer", "null"]}}}})
            self.assertEqual(wt.clean_args({"k": None}), {"k": None})
            self.assertEqual(wt.clean_args({"k": "7"}), {"k": 7})
        finally:
            for s in servers:
                s.stop()

    def test_serve_evals_and_oracle_fixes(self):
        import os, subprocess, threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from unittest import mock
        from harness import evals, serve

        # _version with -cuda12 + _ggufs parent dir check + sorted + LD_LIBRARY_PATH
        self.assertGreater(serve._version(Path("/b/llama.cpp-v1.10.0-cuda12/llama-server")),
                           serve._version(Path("/b/llama.cpp-v1.2.0-cuda12/llama-server")))
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup)
        models = Path(d.name) / ".lmstudio" / "models"
        (models / "BAAI" / "bge-m3-gguf").mkdir(parents=True)
        (models / "BAAI" / "bge-m3-gguf" / "q8_0.gguf").write_bytes(b"0")
        (models / "chat").mkdir(parents=True)
        (models / "chat" / "z.gguf").write_bytes(b"0")
        (models / "chat" / "a.gguf").write_bytes(b"00")
        with mock.patch.object(serve, "LMS", Path(d.name) / ".lmstudio"):
            self.assertEqual(len(serve._ggufs(True)), 1)
            self.assertEqual(serve.find_model("chat").name, "a.gguf")   # sắp xếp xác định thay vì ngẫu nhiên inode
        backends_dir = Path(d.name) / ".lmstudio" / "extensions" / "backends" / "llama.cpp-v1.0.0"
        backends_dir.mkdir(parents=True)
        (backends_dir / "llama-server").write_text("")
        with mock.patch.object(serve, "LMS", Path(d.name) / ".lmstudio"), \
                mock.patch("shutil.which", lambda _: None), \
                mock.patch.dict(os.environ, {"LD_LIBRARY_PATH": ""}, clear=False):
            _, env = serve.find_server()
            self.assertFalse(env["LD_LIBRARY_PATH"].endswith(":"))

        # healthy() checks JSON status == ok
        body_holder = [b"<html>not llama</html>"]
        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_GET(self):
                self.send_response(200); self.end_headers(); self.wfile.write(body_holder[0])
        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close); self.addCleanup(srv.shutdown)
        ls = serve.LocalServer(Path("/m"), port=srv.server_port)
        self.assertFalse(ls.healthy())
        body_holder[0] = b'{"status":"ok"}'
        self.assertTrue(ls.healthy())

        # evals._count_callers + --repeat 0 + oracle.py --list
        self.assertTrue(evals._count_callers(None, ["Có 2 file gọi helper() (tổng 3 lần gọi)."])[0])
        self.assertFalse(evals._count_callers(None, ["Có 3 file gọi helper()."])[0])
        import contextlib
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            evals.main(["--repeat", "0"])
        r_or = subprocess.run([sys.executable, str(Path(__file__).resolve().parent.parent / "examples" / "oracle" / "oracle.py"), "--list"],
                              cwd=d.name, capture_output=True, text=True)
        self.assertEqual(r_or.returncode, 0)
        self.assertEqual(len(json.loads(r_or.stdout)), 7)


class AuditRegressionTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.ws = Workspace(self.d.name)

    def test_tool_clean_args_rejects_bool_as_integer(self):
        from harness.tools import P, Tool
        t = Tool("t", "d", {"n": P("integer", "num")}, lambda **kw: str(kw["n"]))
        self.assertTrue(str(t.clean_args({"n": True})).startswith("LỖI:"))
        self.assertTrue(str(t.clean_args({"n": False})).startswith("LỖI:"))
        self.assertEqual(t.clean_args({"n": 3}), {"n": 3})
        self.assertEqual(t.clean_args({"n": "7"}), {"n": 7})

    def test_agent_context_file_symlink_escape_and_internal_allowed(self):
        out_dir = tempfile.TemporaryDirectory()
        self.addCleanup(out_dir.cleanup)
        secret = Path(out_dir.name) / "secret.txt"
        secret.write_text("TOP_SECRET_LEAK_XYZ")
        (self.ws.root / "HARNESS.md").symlink_to(secret)
        a = Agent(FakeClient([]), build_tools(self.ws), self.ws, ui=UI(io.StringIO()))
        self.assertNotIn("TOP_SECRET_LEAK_XYZ", a.messages[0]["content"])

        (self.ws.root / "HARNESS.md").unlink()
        (self.ws.root / "docs").mkdir()
        (self.ws.root / "docs" / "guide.md").write_text("SAFE_INTERNAL_GUIDE")
        (self.ws.root / "HARNESS.md").symlink_to(Path("docs/guide.md"))
        a.reset()
        self.assertIn("SAFE_INTERNAL_GUIDE", a.messages[0]["content"])

    def test_apply_system_schema_len_zero_headroom_and_compact_budget_cap(self):
        a = Agent(FakeClient([]), build_tools(self.ws), self.ws, ui=UI(io.StringIO()), max_context_tokens=200)
        a._cpt = 5.0
        a._base_system = "S" * 800
        a._schema_len = 500
        a.summary = "SECRET_SUMMARY"
        a._apply_system()
        self.assertEqual(len(a.messages[0]["content"]), 800)

        class SumClient:
            def __init__(self):
                self.n = 0
            def chat(self, messages, tools=None, on_token=None, on_thinking=None):
                if messages and messages[0].get("content", "").startswith("Tóm tắt"):
                    return Reply(content="Z" * 500)
                self.n += 1
                if self.n == 1:
                    return Reply(content="R1 " + ("a" * 350))
                return Reply(content="R2")

        b = Agent(SumClient(), build_tools(self.ws), self.ws, ui=UI(io.StringIO()),
                  max_context_chars=1800, summarize=True)
        b._base_system = "B" * 800
        b.messages[0]["content"] = b._base_system
        b.run("hi1")
        # Bước 2: lượt 1 dài + user 2 (~900) buộc compact; compact KHÔNG được đẩy _size() vượt _budget() qua msg0
        b.messages.append({"role": "user", "content": "U2 " + ("u" * 880), "_turn": 2})
        b.turn = 2
        b.compact()
        self.assertLessEqual(b._size(), b._budget())

    def test_ledger_text_tools_gate_null_function_and_pytest_param(self):
        from harness.agent import fingerprint
        a = Agent(FakeClient([]), build_tools(self.ws), self.ws, ui=UI(io.StringIO()), text_tools=False)
        msg_tc = {"role": "assistant", "content": '<tool_call>{"name": "write_file", "arguments": {"path": "x.py", "content": "a"}}</tool_call>'}
        a._ledger_add([msg_tc])
        self.assertEqual(a.ledger, [])

        a2 = Agent(FakeClient([]), build_tools(self.ws), self.ws, ui=UI(io.StringIO()), text_tools=True)
        a2._ledger_add([msg_tc])
        self.assertEqual(a2.ledger, ["write_file: x.py (1 dòng)"])

        bad_msg = {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function", "function": None}]}
        self.assertEqual(a._msg_len(bad_msg), 0)
        self.assertIn("[gọi tool: ?]", a._transcript([bad_msg]))
        self.assertEqual(fingerprint("tests/t.py::TestA::test_p[a b] FAILED [ 50%]"),
                         frozenset({"tests/t.py::TestA::test_p[a b]"}))

    def test_goal_resume_baselines_mismatch_legacy_and_cli_final_fail(self):
        import contextlib, threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from unittest import mock
        from harness import cli
        from harness.goal import run_goal

        todo = lambda t: Reply(tool_calls=[{"id": "t", "name": "todo", "arguments": {"todos": t}}])
        w = lambda p: Reply(tool_calls=[{"id": p, "name": "write_file", "arguments": {"path": p, "content": "v = 1\n"}}])

        # 1) Baseline frozenset: giữ kiểu frozenset sau resume, vẫn chặn lỗi MỚI và cho qua lỗi CŨ
        outs1 = [
            (False, "FAIL: test_old (m.T.test_old)"),
            (False, "FAIL: test_old (m.T.test_old)"),
            (False, "FAIL: test_old (m.T.test_old)\nFAIL: test_new (m.T.test_new)"),
            (False, "FAIL: test_old (m.T.test_old)\nFAIL: test_new (m.T.test_new)"),
            (False, "FAIL: test_old (m.T.test_old)\nFAIL: test_new (m.T.test_new)"),
        ]
        saved = self.ws.root / "bl_set.json"
        a = Agent(FakeClient([todo("[ ] v1\n[ ] v2"), Reply(content="plan"), w("a.py"), Reply(content="done1")]),
                  build_tools(self.ws), self.ws, ui=UI(io.StringIO()),
                  verify=lambda: outs1.pop(0), verify_rounds=1)
        def stop1():
            if len([1 for it in a.goal_state["items"] if it["status"] != "pending"]) == 1:
                a.save(saved)
                raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            run_goal(a, "gset", step_budget=5, plan_steps=5, require_tests=False, on_item_done=stop1)

        b = Agent(FakeClient([w("b.py"), Reply(content="thử 1"), Reply(content="vẫn còn test_new")]),
                  build_tools(self.ws), self.ws, ui=UI(io.StringIO()),
                  verify=lambda: outs1.pop(0) if outs1 else (False, "FAIL: test_old (m.T.test_old)\nFAIL: test_new (m.T.test_new)"),
                  verify_rounds=1)
        self.assertTrue(b.load(saved))
        res_b = run_goal(b, "gset", step_budget=5, plan_steps=5, require_tests=False)
        self.assertEqual([r.status for r in res_b], ["done", "blocked"])

        # 2) Baseline str + goal mismatch replans fresh
        a2 = Agent(FakeClient([todo("[ ] x1\n[ ] x2"), Reply(content="p"), Reply(content="done1")]),
                   build_tools(self.ws), self.ws, ui=UI(io.StringIO()),
                   verify=lambda: (False, "SyntaxError: bad"))
        saved_str = self.ws.root / "bl_str.json"
        def stop_str():
            a2.save(saved_str)
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            run_goal(a2, "goal A", step_budget=4, plan_steps=4, require_change=False, on_item_done=stop_str)
        self.assertEqual(json.loads(saved_str.read_text())["goal_state"]["baseline"], "SyntaxError: bad")

        # Chạy lại cùng goal -> giữ baseline là str; đổi goal -> lập kế hoạch mới
        c = Agent(FakeClient([todo("[ ] new_task"), Reply(content="p2"), Reply(content="done new")]),
                  build_tools(self.ws), self.ws, ui=UI(io.StringIO()))
        self.assertTrue(c.load(saved_str))
        res_c = run_goal(c, "goal B", step_budget=4, plan_steps=4, require_change=False)
        self.assertEqual([r.text for r in res_c], ["new_task"])

        # 3) Legacy todo-only resume (không có goal_state)
        leg = self.ws.root / "legacy.json"
        a_leg = Agent(FakeClient([]), build_tools(self.ws), self.ws, ui=UI(io.StringIO()))
        a_leg.tools["todo"].fn("[x] mục cũ\n[ ] mục hai")
        a_leg.save(leg)
        d_leg = Agent(FakeClient([Reply(content="xong 2")]), build_tools(self.ws), self.ws, ui=UI(io.StringIO()))
        self.assertTrue(d_leg.load(leg))
        res_leg = run_goal(d_leg, "any", step_budget=4, plan_steps=4, require_change=False)
        self.assertEqual([(r.text, r.status) for r in res_leg], [("mục cũ", "done"), ("mục hai", "done")])

        # 4) --resume file hỏng JSON hoặc sai kiểu bị chặn TRƯỚC backends(); --goal final_ok False -> exit 2
        sess_dir = self.ws.root / ".harness" / "sessions"
        sess_dir.mkdir(parents=True, exist_ok=True)
        (sess_dir / "corrupt.json").write_text("{not json")
        (sess_dir / "badbool.json").write_text('{"messages": [], "turn": true, "first_live_turn": 1}')
        called = []
        @contextlib.contextmanager
        def spy_backends(args):
            called.append(1)
            yield args
        with mock.patch.object(cli, "backends", spy_backends), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["-C", self.d.name, "--resume", "corrupt"]), 1)
            self.assertEqual(cli.main(["-C", self.d.name, "--resume", "badbool"]), 1)
        self.assertEqual(called, [])

        # Mock LLM server cho cli.main --goal (items done nhưng final verify exit 1 -> rc=2)
        calls = [0]
        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_GET(self):
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
                self.wfile.write(json.dumps({"data": [{"id": "fake"}]}).encode())
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                calls[0] += 1
                self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
                if calls[0] == 1:
                    tc = {"index": 0, "id": "t1", "function": {"name": "todo", "arguments": json.dumps({"todos": "[ ] tạo f"})}}
                    self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {'tool_calls': [tc]}}]})}\n\n".encode())
                elif calls[0] == 3:
                    tc1 = {"index": 0, "id": "w1", "function": {"name": "write_file", "arguments": json.dumps({"path": "f.py", "content": "x=1\n"})}}
                    tc2 = {"index": 1, "id": "w2", "function": {"name": "write_file", "arguments": json.dumps({"path": "test_f.py", "content": "assert True\n"})}}
                    self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {'tool_calls': [tc1, tc2]}}]})}\n\n".encode())
                else:
                    self.wfile.write(b'data: {"choices":[{"delta":{"content":"xong"}}]}\n\n')
                self.wfile.write(b"data: [DONE]\n\n")
        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close); self.addCleanup(srv.shutdown)
        gdir = tempfile.TemporaryDirectory(); self.addCleanup(gdir.cleanup)
        out = io.StringIO()
        with mock.patch.object(cli, "load_mcp_tools", lambda *a, **k: ({}, [])), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            rc = cli.main(["--host", f"http://127.0.0.1:{srv.server_port}", "-m", "fake", "-C", gdir.name,
                           "--no-memory", "--no-embed", "--no-sandbox", "--verify", "exit 1", "--goal", "làm f"])
        self.assertEqual(rc, 2)

    def test_cli_llm_error_cleanup_repl_and_headless(self):
        import contextlib, threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from unittest import mock
        from harness import cli

        (self.ws.root / "note.txt").write_text("ok")
        req_n = [0]
        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_GET(self):
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
                self.wfile.write(json.dumps({"data": [{"id": "fake"}]}).encode())
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                req_n[0] += 1
                if req_n[0] in (2, 5):  # bước 2 (sau tool) trả HTTP 400 (non-transient LLMError)
                    self.send_response(400); self.end_headers(); self.wfile.write(b"bad request"); return
                self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
                if req_n[0] in (1, 4):
                    tc = {"index": 0, "id": "r1", "function": {"name": "read_file", "arguments": json.dumps({"path": "note.txt"})}}
                    self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {'tool_calls': [tc]}}]})}\n\n".encode())
                else:
                    self.wfile.write(b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n')
                self.wfile.write(b"data: [DONE]\n\n")
        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close); self.addCleanup(srv.shutdown)
        out = io.StringIO()
        # REPL: lượt 1 lỗi sau khi tool đã chạy, lượt 2 vẫn hợp lệ
        with mock.patch.object(cli, "load_mcp_tools", lambda *a, **k: ({}, [])), \
                mock.patch("builtins.input", side_effect=["đọc thử", "hỏi tiếp", "/exit"]), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            rc_repl = cli.main(["--host", f"http://127.0.0.1:{srv.server_port}", "-m", "fake", "-C", self.d.name,
                                "--no-memory", "--no-embed", "--no-sandbox"])
        self.assertEqual(rc_repl, 0)
        saved_repl = json.loads((self.ws.root / ".harness" / "sessions" / "last.json").read_text())
        roles = [m["role"] for m in saved_repl["messages"]]
        for i in range(len(roles) - 1):
            self.assertFalse(roles[i] == "tool" and roles[i + 1] == "user")

        # Headless -p: lỗi sau tool -> cleanup_interrupted() trước khi lưu
        d2 = tempfile.TemporaryDirectory(); self.addCleanup(d2.cleanup)
        (Path(d2.name) / "note.txt").write_text("ok")
        with mock.patch.object(cli, "load_mcp_tools", lambda *a, **k: ({}, [])), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            rc_p = cli.main(["--host", f"http://127.0.0.1:{srv.server_port}", "-m", "fake", "-C", d2.name,
                             "--no-memory", "--no-embed", "--no-sandbox", "-p", "chạy"])
        self.assertEqual(rc_p, 1)
        saved_p = json.loads((Path(d2.name) / ".harness" / "sessions" / "last.json").read_text())
        self.assertEqual(saved_p["messages"][-1]["role"], "assistant")

    def test_mcp_bad_json_types_and_anyof_cleanup(self):
        from harness.mcp import _wrap, load_mcp_tools
        (self.ws.root / ".harness").mkdir(exist_ok=True)
        warns = []
        (self.ws.root / ".harness" / "mcp.json").write_text("[]")
        tools, srvs = load_mcp_tools(self.ws.root, warn=warns.append)
        self.assertEqual((tools, srvs), ({}, []))
        self.assertTrue(warns)

        for bad_cfg in [
            {"mcpServers": {"s": {"command": 123}}},
            {"mcpServers": {"s": {"command": ""}}},
            {"mcpServers": {"s": {"command": "echo", "args": "not-list"}}},
            {"mcpServers": {"s": {"command": "echo", "env": ["bad"]}}},
        ]:
            warns.clear()
            (self.ws.root / ".harness" / "mcp.json").write_text(json.dumps(bad_cfg))
            self.assertEqual(load_mcp_tools(self.ws.root, warn=warns.append), ({}, []))
            self.assertTrue(warns)

        class DummySrv:
            name = "d"
            def call(self, tool, kw): return "ok"
        wrapped = _wrap(DummySrv(), {
            "name": "fn",
            "inputSchema": {"properties": {"n": {"anyOf": [{"type": "integer"}, {"type": "null"}]}}, "required": ["n"]},
        })
        self.assertEqual(wrapped.parameters["n"]["type"], "integer")
        self.assertTrue(wrapped.parameters["n"]["nullable"])
        self.assertNotIn("anyOf", wrapped.schema()["function"]["parameters"]["properties"]["n"])

    def test_client_http_errors_ctx_null_error_bool_index_and_serve_stop(self):
        import gc, subprocess, threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from harness import serve
        from harness.client import LLMError, LMStudioClient, _EMBED_RX
        from harness.embed import EmbedClient, EmbedError

        mode = ["ok"]
        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_GET(self):
                if mode[0] == "500":
                    self.send_response(500); self.end_headers(); self.wfile.write(b"err"); return
                if mode[0] == "badjson":
                    self.send_response(200); self.end_headers(); self.wfile.write(b"{bad"); return
                if self.path == "/api/v0/models" and mode[0] == "props":
                    self.send_response(404); self.end_headers(); return
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
                if self.path == "/api/v0/models":
                    self.wfile.write(json.dumps({"data": [{"id": "m1", "state": "loaded", "loaded_context_length": 4096}]}).encode())
                elif self.path == "/props":
                    self.wfile.write(json.dumps({"default_generation_settings": {"n_ctx": 8192}}).encode())
                else:
                    self.wfile.write(json.dumps({"data": [{"id": "m1"}]}).encode())
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                if mode[0] == "500":
                    self.send_response(500); self.end_headers(); self.wfile.write(b"err"); return
                self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
                tc = {"index": True, "id": "c1", "function": {"name": "read_file", "arguments": '{"path":"a"}'}}
                self.wfile.write(f"data: {json.dumps({'error': None, 'choices': [{'delta': {'tool_calls': [tc]}}]})}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close); self.addCleanup(srv.shutdown)

        c = LMStudioClient("m1", f"http://127.0.0.1:{srv.server_port}")
        self.assertEqual(c.context_length(), 4096)
        mode[0] = "props"
        self.assertEqual(c.context_length(), 8192)
        mode[0] = "ok"
        rep = c.chat([{"role": "user", "content": "hi"}], None)
        self.assertEqual(rep.tool_calls, [{"id": "c1", "name": "read_file", "arguments": {"path": "a"}}])
        mode[0] = "500"
        with self.assertRaises(LLMError):
            c.list_models()
        self.assertFalse(serve.LocalServer(Path("/m"), port=srv.server_port).healthy())
        with self.assertRaises(EmbedError):
            EmbedClient(f"http://127.0.0.1:{srv.server_port}").embed(["hi"])
        mode[0] = "badjson"
        with self.assertRaises(LLMError):
            c.list_models()
        self.assertIsNone(c.context_length())
        gc.collect()

        self.assertRegex("bge_m3.gguf", _EMBED_RX)
        self.assertRegex("e5_large.gguf", serve.EMBED_RX)
        self.assertIsNone(_EMBED_RX.search("qwen2.5-7b.gguf"))

        ls = serve.LocalServer(Path("/m"))
        waits = []
        class FakeProc:
            def poll(self): return None
            def terminate(self): pass
            def wait(self, timeout=None):
                waits.append(timeout)
                if len(waits) == 1:
                    raise subprocess.TimeoutExpired("cmd", timeout)
                return 0
            def kill(self): waits.append("kill")
        ls.proc = FakeProc()
        ls.stop()
        self.assertEqual(waits, [10, "kill", 2])

    def test_bash_landlock_symlink_then_tools_blocked(self):
        from harness import sandbox
        if not sandbox.available():
            self.skipTest("Landlock không bật trên kernel này")
        out_dir = tempfile.TemporaryDirectory()
        self.addCleanup(out_dir.cleanup)
        secret = Path(out_dir.name) / "secret.txt"
        secret.write_text("OUTSIDE_SECRET_LINE\n")
        tools = build_tools(self.ws)
        tools["bash"].fn(f"ln -s '{secret}' link.txt")
        self.assertTrue((self.ws.root / "link.txt").is_symlink())
        with self.assertRaises(ValueError):
            tools["read_file"].fn("link.txt")
        self.assertNotIn("OUTSIDE_SECRET_LINE", tools["grep"].fn("OUTSIDE_SECRET_LINE"))
        self.assertNotIn("link.txt", tools["glob"].fn("*"))

    def test_oracle_runtime_stage1_and_c3_ask(self):
        import subprocess
        app_root = Path(self.d.name)
        pkg = app_root / "people_memory"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "__main__.py").write_text(
            "import json, os, sqlite3, urllib.request\n"
            "from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer\n"
            "from pathlib import Path\n"
            "data_dir = Path(os.environ['PM_DATA_DIR']); data_dir.mkdir(parents=True, exist_ok=True)\n"
            "db = sqlite3.connect(data_dir / 'people.db', check_same_thread=False)\n"
            "db.execute('CREATE TABLE IF NOT EXISTS people (id INTEGER PRIMARY KEY, name TEXT, relation TEXT)')\n"
            "db.execute('CREATE TABLE IF NOT EXISTS facts (id INTEGER PRIMARY KEY, pid INT, text TEXT, key TEXT, kind TEXT, status TEXT, superseded_at TEXT)')\n"
            "db.commit()\n"
            "def _llm(msgs):\n"
            "    req = urllib.request.Request(os.environ['PM_LLM_URL'].rstrip('/') + '/chat/completions',\n"
            "        data=json.dumps({'model': 'm', 'messages': msgs}).encode(), headers={'Content-Type': 'application/json'})\n"
            "    with urllib.request.urlopen(req, timeout=5) as r:\n"
            "        return json.loads(json.load(r)['choices'][0]['message']['content'])\n"
            "class H(BaseHTTPRequestHandler):\n"
            "    def log_message(self, *a): pass\n"
            "    def _send(self, code, obj, ct='application/json'):\n"
            "        body = obj.encode() if isinstance(obj, str) else json.dumps(obj).encode()\n"
            "        self.send_response(code); self.send_header('Content-Type', ct); self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)\n"
            "    def do_GET(self):\n"
            "        if self.path == '/api/health': return self._send(200, {'ok': True})\n"
            "        if self.path == '/': return self._send(200, '<html><body><form><input name=\"q\"><button>Send</button></form></body></html>', 'text/html')\n"
            "        if self.path == '/api/people':\n"
            "            rows = db.execute('SELECT id, name, relation FROM people').fetchall()\n"
            "            return self._send(200, {'people': [{'id': r[0], 'name': r[1], 'relation': r[2], 'aliases': []} for r in rows]})\n"
            "        if self.path.startswith('/api/people/'):\n"
            "            pid = int(self.path.rsplit('/', 1)[-1])\n"
            "            p = db.execute('SELECT id, name, relation FROM people WHERE id=?', (pid,)).fetchone()\n"
            "            fs = [{'text': r[0], 'key': r[1], 'kind': r[2], 'status': r[3], 'superseded_at': r[4]} for r in db.execute('SELECT text, key, kind, status, superseded_at FROM facts WHERE pid=?', (pid,)).fetchall()]\n"
            "            return self._send(200, {'id': p[0], 'name': p[1], 'relation': p[2], 'aliases': [], 'facts': fs})\n"
            "        self._send(404, {'error': 'not found'})\n"
            "    def do_POST(self):\n"
            "        raw = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')\n"
            "        if self.path == '/api/message':\n"
            "            res = _llm([{'role': 'user', 'content': raw['text']}])\n"
            "            for p in res.get('people', []):\n"
            "                row = db.execute('SELECT id FROM people WHERE name=?', (p['name'],)).fetchone()\n"
            "                pid = row[0] if row else db.execute('INSERT INTO people(name, relation) VALUES (?,?)', (p['name'], p.get('relation'))).lastrowid\n"
            "                for f in p.get('facts', []):\n"
            "                    if f.get('key'): db.execute(\"UPDATE facts SET status='superseded', superseded_at='now' WHERE pid=? AND key=? AND status='active'\", (pid, f['key']))\n"
            "                    db.execute(\"INSERT INTO facts(pid, text, key, kind, status) VALUES (?,?,?,?,'active')\", (pid, f['text'], f.get('key'), f.get('kind')))\n"
            "            db.commit()\n"
            "            return self._send(200, {'status': 'saved'})\n"
            "        if self.path == '/api/ask':\n"
            "            q = raw.get('question', '')\n"
            "            row = next((r for r in db.execute('SELECT id, name FROM people').fetchall() if r[1] in q), None)\n"
            "            fs = [{'text': r[0], 'status': 'active'} for r in db.execute(\"SELECT text FROM facts WHERE pid=? AND status='active'\", (row[0],)).fetchall()] if row else []\n"
            "            return self._send(200, {'answer': ', '.join(f['text'] for f in fs) or 'chưa có thông tin', 'facts': fs, 'people': [row[1]] if row else []})\n"
            "        self._send(404, {'error': 'not found'})\n"
            "ThreadingHTTPServer((os.environ.get('PM_HOST', '127.0.0.1'), int(os.environ['PM_PORT'])), H).serve_forever()\n"
        )
        oracle_py = str(Path(__file__).resolve().parent.parent / "examples" / "oracle" / "oracle.py")
        r1 = subprocess.run([sys.executable, "-W", "error::ResourceWarning", oracle_py, "--root", str(app_root), "--stage", "1"],
                            capture_output=True, text=True, timeout=30)
        self.assertEqual(r1.returncode, 0, r1.stdout + r1.stderr)
        r3 = subprocess.run([sys.executable, "-W", "error::ResourceWarning", oracle_py, "--root", str(app_root),
                             "--stage", "3", "--only", "trả fact ACTIVE"],
                            capture_output=True, text=True, timeout=30)
        self.assertEqual(r3.returncode, 0, r3.stdout + r3.stderr)


if __name__ == "__main__":
    unittest.main()

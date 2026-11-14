import io
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()


class MoreTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
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
        self.ws = Workspace(self.d.name)
        from harness.memory import Memory, memory_tools
        self.mem = Memory(":memory:", "s1")
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
        d = tempfile.TemporaryDirectory(); ws = Workspace(d.name)
        a = Agent(FakeClient([Reply(content="x" * 200)] * 5), build_tools(ws), ws, ui=UI(io.StringIO()),
                  memory=Memory(":memory:", "s"), max_context_chars=1300)
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
        m = Memory(":memory:", "s", FakeEmbedder())
        m.add("Mật khẩu wifi là ZEBRA42", "turn", 1)
        m.add("Bài thơ mùa thu rất hay", "turn", 2)
        self.assertEqual(m.search("thông tin đăng nhập", k=1)[0].content, "Mật khẩu wifi là ZEBRA42")  # BM25 một mình sẽ trượt
        self.assertEqual(Memory(":memory:", "s").search("thông tin đăng nhập"), [])

    def test_embed_failure_falls_back_to_bm25(self):
        from harness.embed import EmbedError
        from harness.memory import Memory

        class Broken:
            def embed(self, texts):
                raise EmbedError("down")
        m = Memory(":memory:", "s", Broken())
        m.add("Mật khẩu wifi là ZEBRA42", "turn", 1)
        self.assertEqual(len(m.search("mat khau")), 1)
        self.assertEqual(m.stats().get("vector"), 0)

    def test_backfill(self):
        from harness.memory import Memory
        m = Memory(":memory:", "s")
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
        d = tempfile.TemporaryDirectory(); ws = Workspace(d.name)
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
        d = tempfile.TemporaryDirectory()
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
        d = tempfile.TemporaryDirectory(); ws = Workspace(d.name)
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
        a = self.agent(c, memory=Memory(":memory:", "s"), summarize=True, max_context_chars=1300)
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
        a = self.agent(SmartClient([Reply(content="đáp 1")]), memory=Memory(":memory:", "phien-A"))
        a.run("câu 1")
        a.summary = "- tóm tắt"
        a.save(self.ws.root / "s.json")
        m2 = Memory(":memory:", "phien-B")
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
        d = tempfile.TemporaryDirectory()
        from pathlib import Path
        (Path(d.name) / "settings.ini").write_text("[server]\nhost = 0.0.0.0\nport = 9090\ndebug = false\n")
        self.assertTrue(_precise_edit(Path(d.name), [])[0])

    def test_compare(self):
        import json
        from pathlib import Path
        from harness.evals import compare
        d = tempfile.TemporaryDirectory()
        f = Path(d.name) / "old.json"
        f.write_text(json.dumps({"label": "cũ", "results": [{"task": "t", "passed": False}, {"task": "t", "passed": True}]}))
        out = compare(str(f), [{"task": "t", "passed": True}, {"task": "t", "passed": True}])
        self.assertIn("1/2 → 2/2 ▲", out)


class SearchBackendTests(unittest.TestCase):
    def test_format_searx(self):
        from harness.tools import format_searx
        out = format_searx({"results": [{"title": "A", "url": "https://a.io", "content": "mô tả"}, {"title": "B", "url": "https://b.io"}]})
        self.assertEqual(out, "1. A\n   https://a.io\n   mô tả\n2. B\n   https://b.io")

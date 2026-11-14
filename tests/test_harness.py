import io
import json
from pathlib import Path
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


@unittest.skipUnless(__import__("harness.sandbox", fromlist=["x"]).available(), "kernel không có Landlock")
class SandboxTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
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
        d = tempfile.TemporaryDirectory(); ws = Workspace(d.name)
        out = build_tools(ws)["bash"].fn("cat /etc/passwd")
        self.assertIn("[sandbox]", out)
        self.assertIn("TƯƠNG ĐỐI", out)
        self.assertNotIn("[sandbox]", build_tools(ws)["bash"].fn("echo ok"))
        a = Agent(FakeClient([]), build_tools(ws), ws, ui=UI(io.StringIO()))
        self.assertIn("đường dẫn TƯƠNG ĐỐI", a.messages[0]["content"])


class PromptHidesAbsolutePathTest(unittest.TestCase):
    def test_no_absolute_root_in_system_prompt(self):
        d = tempfile.TemporaryDirectory(); ws = Workspace(d.name)
        sys_prompt = Agent(FakeClient([]), build_tools(ws), ws, ui=UI(io.StringIO())).messages[0]["content"]
        self.assertNotIn(str(ws.root), sys_prompt)   # model hay gõ sai đường dẫn tuyệt đối
        self.assertIn("thư mục hiện tại", sys_prompt)


class LongTurnCompactionTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory(); self.ws = Workspace(self.d.name)

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
        self.d = tempfile.TemporaryDirectory(); self.ws = Workspace(self.d.name)

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
        d = tempfile.TemporaryDirectory(); ws = Workspace(d.name)
        t = build_tools(ws)
        t["write_file"].fn("c.py", "def add(a, b):\n    return a - b\n\nprint(add(1, 2))\n")
        out = t["edit_file"].fn("c.py", "  return a -b", "  return a + b")
        self.assertIn("2\t    return a - b", out)       # chép lại đúng dòng thật, có số dòng
        self.assertIn("không tìm thấy", t["edit_file"].fn("c.py", "hoàn toàn khác biệt xyz 123", "q"))


class ToolCallArgsCountedTests(unittest.TestCase):
    def test_write_file_args_count_and_shrink(self):
        d = tempfile.TemporaryDirectory(); ws = Workspace(d.name)
        a = Agent(SmartClient([]), build_tools(ws), ws, ui=UI(io.StringIO()), max_context_chars=2500)
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
        d = tempfile.TemporaryDirectory(); ws = Workspace(d.name); t = build_tools(ws)
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
        d = tempfile.TemporaryDirectory(); ws = Workspace(d.name)
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
        self.d = tempfile.TemporaryDirectory(); self.ws = Workspace(self.d.name)

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
        d = tempfile.TemporaryDirectory(); ws = Workspace(d.name)
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
        d = tempfile.TemporaryDirectory(); ws = Workspace(d.name)
        a = Agent(SmartClient([Reply(content="xong")] * 4), build_tools(ws), ws, ui=UI(io.StringIO()),
                  verify=lambda: (False, "FAIL: test_x (m.T.test_x)\nFAILED (failures=1)"), verify_rounds=3)
        a.run("việc")
        msgs = [m["content"] for m in a.messages if "[Kiểm tra tự động]" in (m.get("content") or "")]
        self.assertNotIn("GIỐNG HỆT", msgs[0])      # lần đầu chưa gợi ý
        self.assertIn("GIỐNG HỆT", msgs[1])         # lặp lại thì gợi ý xét chính test
        self.assertIn("ngày giờ cố định", msgs[1])


class HysteresisTests(unittest.TestCase):
    def test_compacts_once_then_stays_stable(self):
        d = tempfile.TemporaryDirectory(); ws = Workspace(d.name)
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
        d = tempfile.TemporaryDirectory(); ws = Workspace(d.name)
        loop = [Reply(tool_calls=[{"id": str(i), "name": "list_dir", "arguments": {"path": "."}}]) for i in range(3)]
        a = Agent(SmartClient(loop), build_tools(ws), ws, ui=UI(io.StringIO()), max_steps=3)
        a.run("x")
        self.assertTrue(a.exhausted)
        b = Agent(SmartClient([Reply(content="")] * 3), build_tools(ws), ws, ui=UI(io.StringIO()))
        b.run("x")                                       # kết thúc tự nhiên nhưng câu trả lời cuối RỖNG
        self.assertFalse(b.exhausted)


class GoalPlanOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory(); self.ws = Workspace(self.d.name)

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
        self.d = tempfile.TemporaryDirectory(); self.ws = Workspace(self.d.name)

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
        run_goal(a, "mục tiêu", step_budget=6, plan_steps=5)
        self.assertEqual(len(calls), 4)   # 1 baseline + 1 sau việc A (test_moi là lỗi MỚI nên bị chặn) + 1 sau vòng sửa + 1 kiểm tra cuối; KHÔNG chạy khi lập kế hoạch
        self.assertTrue(any("[Kiểm tra tự động]" in (m.get("content") or "") and "test_moi" in m["content"] for m in a.messages))


class LedgerTests(unittest.TestCase):
    def test_evicted_actions_stay_visible_in_system_prompt(self):
        d = tempfile.TemporaryDirectory(); ws = Workspace(d.name)
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
        d = tempfile.TemporaryDirectory(); ws = Workspace(d.name)
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
        try:
            r = LMStudioClient("m", f"http://127.0.0.1:{srv.server_port}").chat([{"role": "user", "content": "x"}])
        finally:
            srv.shutdown()
        self.assertEqual(r.aborted, "repetition")
        self.assertLess(len(r.thinking), 2000)     # ngắt sớm, không đọc hết 400 chunk

    def test_agent_nudges_after_aborted_reply_and_continues(self):
        d = tempfile.TemporaryDirectory(); ws = Workspace(d.name)
        stuck = Reply(content="", aborted="repetition")
        a = Agent(SmartClient([stuck, Reply(content="đã chọn một hành động và xong")]), build_tools(ws), ws, ui=UI(io.StringIO()))
        self.assertEqual(a.run("việc"), "đã chọn một hành động và xong")
        self.assertTrue(any("[Hệ thống]" in (m.get("content") or "") and "MỘT hành động" in m["content"] for m in a.messages))


class DocOnlyItemTests(unittest.TestCase):
    def test_readme_item_does_not_need_test_file(self):
        from harness.goal import run_goal
        d = tempfile.TemporaryDirectory(); ws = Workspace(d.name)
        todo = Reply(tool_calls=[{"id": "t", "name": "todo", "arguments": {"todos": "[ ] viết README ghi test nào chứng minh"}}])
        w = Reply(tool_calls=[{"id": "w", "name": "write_file", "arguments": {"path": "README.md", "content": "# x\n"}}])
        a = Agent(SmartClient([todo, Reply(content="kế hoạch"), w, Reply(content="xong")]), build_tools(ws), ws, ui=UI(io.StringIO()))
        self.assertEqual([r.status for r in run_goal(a, "mục tiêu", step_budget=4, plan_steps=5)], ["done"])


class EmptyReplyTests(unittest.TestCase):
    def test_empty_reply_is_retried_with_nudge(self):
        d = tempfile.TemporaryDirectory(); ws = Workspace(d.name)
        a = Agent(SmartClient([Reply(content=""), Reply(content="  "), Reply(content="đã làm xong")]), build_tools(ws), ws,
                  ui=UI(io.StringIO()))
        self.assertEqual(a.run("việc"), "đã làm xong")
        self.assertEqual(sum("[Hệ thống] Bạn chưa trả lời gì" in (m.get("content") or "") for m in a.messages), 2)

    def test_gives_up_after_two_retries(self):
        d = tempfile.TemporaryDirectory(); ws = Workspace(d.name)
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
        d = tempfile.TemporaryDirectory(); ws = Workspace(d.name)
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

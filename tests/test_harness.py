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

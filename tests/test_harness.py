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

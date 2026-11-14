"""Vòng lặp agent: model -> tool call -> kết quả -> model ... cho đến khi trả lời xong."""
from __future__ import annotations

import json
import re
import time
from typing import Callable

from .client import LMStudioClient
from .tools import Tool, Workspace, clip
from .ui import UI

SYSTEM = """Bạn là trợ lý lập trình chạy trong terminal của người dùng.
Workspace: {root}
Quy tắc:
- Dùng tool để xem file thật trước khi sửa; không đoán nội dung.
- Sửa nhỏ, chính xác (edit_file) thay vì ghi đè cả file.
- Sau khi sửa, chạy lệnh kiểm tra nếu có thể.
- Trả lời ngắn gọn, cùng ngôn ngữ với người dùng."""

TEXT_TOOLS_PROMPT = """
Để gọi tool, xuất đúng định dạng (mỗi lần một tool) rồi dừng:
<tool_call>{{"name": "<tên>", "arguments": {{...}}}}</tool_call>
Tool có sẵn:
{specs}"""

_TOOL_CALL_RX = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_text_tool_calls(text: str) -> list[dict]:
    calls = []
    for m in _TOOL_CALL_RX.finditer(text):
        try:
            d = json.loads(m.group(1))
            calls.append({"id": f"text_{len(calls)}", "name": d["name"], "arguments": d.get("arguments", {})})
        except (json.JSONDecodeError, KeyError):
            continue
    return calls


class Agent:
    def __init__(self, client: LMStudioClient, tools: dict[str, Tool], ws: Workspace,
                 auto_approve: bool = False, text_tools: bool = False, max_steps: int = 25,
                 approve: Callable[[str, dict], bool] | None = None,
                 ui: UI | None = None):
        self.client, self.tools, self.ws = client, tools, ws
        self.auto_approve, self.text_tools, self.max_steps = auto_approve, text_tools, max_steps
        self.approve = approve or self._ask
        self.ui = ui or UI()
        self.messages: list[dict] = []
        self.reset()

    def reset(self):
        system = SYSTEM.format(root=self.ws.root)
        if self.text_tools:
            specs = "\n".join(json.dumps(t.schema()["function"], ensure_ascii=False) for t in self.tools.values())
            system += TEXT_TOOLS_PROMPT.format(specs=specs)
        self.messages = [{"role": "system", "content": system}]

    def _ask(self, name: str, args: dict) -> bool:
        if self.auto_approve:
            return True
        return input(self.ui.ask_approval(name, args)).strip().lower() in ("y", "yes", "c", "co", "có")

    def _execute(self, name: str, args: dict) -> str:
        tool = self.tools.get(name)
        if not tool:
            return f"LỖI: không có tool `{name}`. Có: {', '.join(self.tools)}"
        if tool.needs_approval and not self.approve(name, args):
            return "Người dùng từ chối thao tác này."
        try:
            return tool.fn(**args)
        except TypeError as e:
            return f"LỖI tham số: {e}"
        except Exception as e:  # tool lỗi không được làm sập agent
            return f"LỖI: {type(e).__name__}: {e}"

    def run(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        schemas = None if self.text_tools else [t.schema() for t in self.tools.values()]
        final, t0, steps = "", time.time(), 0
        for _ in range(self.max_steps):
            steps += 1
            self.ui.spinner_start()
            try:
                reply = self.client.chat(self.messages, schemas,
                                         on_token=self.ui.stream_answer, on_thinking=self.ui.stream_think)
            finally:
                self.ui.end_stream()
            calls = reply.tool_calls or parse_text_tool_calls(reply.content)
            msg: dict = {"role": "assistant", "content": reply.content}
            if reply.tool_calls:
                msg["tool_calls"] = [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                    for c in reply.tool_calls]
            self.messages.append(msg)
            if not calls:
                final = reply.content
                break
            for c in calls:
                self.ui.tool_call(c["name"], c["arguments"])
                result = self._execute(c["name"], c["arguments"])
                self.ui.tool_result(result)
                if self.text_tools:  # không dùng tool role native: đưa kết quả như tin nhắn user
                    self.messages.append({"role": "user", "content": f"<tool_result>\n{result}\n</tool_result>"})
                else:
                    self.messages.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": result})
        else:
            self.ui.error(f"dừng: quá {self.max_steps} bước")
        self.ui.footer(time.time() - t0, steps)
        return final

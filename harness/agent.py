"""Vòng lặp agent: model -> tool call -> kết quả -> model ... cho đến khi trả lời xong."""
from __future__ import annotations

import json
import re
import platform
import time
from datetime import date
from typing import Callable

from .client import LMStudioClient
from .memory import Memory, format_hits
from .tools import Tool, Workspace, clip
from .ui import UI

SYSTEM = """Bạn là trợ lý lập trình chạy trong terminal của người dùng.
Workspace: {root} · Hệ điều hành: {os} · Hôm nay: {today}
Quy tắc:
- Suy nghĩ ngắn gọn. Câu hỏi đơn giản thì trả lời thẳng, không cần tool.
- Xem file thật (read_file/grep/glob) trước khi sửa; không đoán nội dung.
- Việc nhiều bước: ghi kế hoạch bằng tool todo rồi cập nhật khi xong từng bước.
- Sửa nhỏ, chính xác bằng edit_file thay vì ghi đè cả file.
- Sau khi sửa, chạy lệnh kiểm tra nếu có thể. Không lặp lại cùng một lệnh khi đã có kết quả.
- Trả lời ngắn gọn, cùng ngôn ngữ với người dùng, dùng markdown tối thiểu."""

MEMORY_PROMPT = """
- Bạn có bộ nhớ ngoài. Hội thoại cũ và kết quả tool cũ được cất ở đó và KHÔNG còn trong context. Khi cần chi tiết đã nói/làm trước đó, gọi recall; việc đáng nhớ lâu dài thì gọi remember. Mục "Ký ức liên quan" (nếu có) là thứ hệ thống tự tìm giúp bạn, chỉ dùng khi hữu ích."""

CONTEXT_FILES = ("HARNESS.md", "AGENTS.md", "CLAUDE.md")

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
                 system_extra: str = "", max_context_chars: int = 40000,
                 ui: UI | None = None, memory: Memory | None = None):
        self.client, self.tools, self.ws = client, tools, ws
        self.auto_approve, self.text_tools, self.max_steps = auto_approve, text_tools, max_steps
        self.approve = approve or self._ask
        self.ui = ui or UI()
        self.system_extra, self.max_context_chars = system_extra, max_context_chars
        self.memory = memory
        self.turn, self.first_live_turn, self._recalled = 0, 1, ""
        self.messages: list[dict] = []
        self.reset()

    def reset(self):
        system = SYSTEM.format(root=self.ws.root, os=platform.system(), today=date.today().isoformat())
        if self.memory:
            system += MEMORY_PROMPT
        self.first_live_turn = self.turn + 1  # các lượt trước /clear chỉ còn trong bộ nhớ ngoài
        for name in CONTEXT_FILES:
            f = self.ws.root / name
            if f.is_file():
                system += f"\n\nHướng dẫn của dự án ({name}):\n{clip(f.read_text(errors='replace'), 4000)}"
                break
        if self.system_extra:
            system += f"\n\n{self.system_extra}"
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
        cleaned = tool.clean_args(args)
        if isinstance(cleaned, str):
            return cleaned
        if tool.needs_approval and not self.approve(name, cleaned):
            return "Người dùng từ chối thao tác này."
        try:
            return tool.fn(**cleaned)
        except Exception as e:  # tool lỗi không được làm sập agent
            return f"LỖI: {type(e).__name__}: {e}"

    def _size(self) -> int:
        return sum(len(m.get("content") or "") for m in self.messages)

    def compact(self) -> int:
        """Giữ context dưới ngân sách. Bước 1: lược kết quả tool cũ. Bước 2: đẩy cả lượt cũ ra bộ nhớ ngoài
        (nội dung đã được lưu khi xảy ra). Luôn giữ 2 lượt gần nhất. Trả về số mục đã giải phóng."""
        n = 0
        for m in self.messages[1:-6]:
            if self._size() <= self.max_context_chars:
                break
            c = m.get("content") or ""
            is_result = m["role"] == "tool" or c.startswith("<tool_result>")
            if is_result and len(c) > 200 and not c.startswith("[đã lược"):
                m["content"] = f"[đã lược {len(c)} ký tự kết quả cũ" + (" — dùng recall để xem lại]" if self.memory else "]")
                n += 1
        if self.memory:
            while self._size() > self.max_context_chars:
                starts = [i for i, m in enumerate(self.messages) if "_turn" in m]
                if len(starts) <= 2:
                    break
                del self.messages[starts[0]:starts[1]]
                self.first_live_turn = self.messages[starts[0]]["_turn"]
                n += 1
        return n

    def _view(self) -> list[dict]:
        """Bản gửi cho model: bỏ khoá nội bộ, và gắn ký ức tự tìm vào lượt hỏi hiện tại (không lưu vào lịch sử)."""
        out = []
        for m in self.messages:
            c = {k: v for k, v in m.items() if not k.startswith("_")}
            if self._recalled and m.get("role") == "user" and m.get("_turn") == self.turn:
                c["content"] = (f"[Ký ức liên quan từ bộ nhớ ngoài — chỉ dùng nếu hữu ích]\n{self._recalled}\n"
                                f"[Hết ký ức]\n\n{m['content']}")
            out.append(c)
        return out

    def run(self, user_input: str) -> str:
        self.turn += 1
        self._recalled = ""
        self.messages.append({"role": "user", "content": user_input, "_turn": self.turn})
        if self.memory:
            self.compact()  # dọn trước, để first_live_turn phản ánh đúng những gì còn trong context
            hits = self.memory.search(user_input, k=4, first_live_turn=self.first_live_turn)
            self._recalled = format_hits(hits)
            if hits:
                self.ui.memory_note(len(hits))
        schemas = None if self.text_tools else [t.schema() for t in self.tools.values()]
        final, t0, steps, seen = "", time.time(), 0, {}
        for _ in range(self.max_steps):
            steps += 1
            self.compact()
            self.ui.spinner_start()
            try:
                reply = self.client.chat(self._view(), schemas,
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
                key = c["name"] + json.dumps(c["arguments"], sort_keys=True, ensure_ascii=False)
                seen[key] = seen.get(key, 0) + 1
                if seen[key] >= 4:
                    result = "LỖI: đã gọi cùng một tool với cùng tham số 4 lần. Dừng lặp: đổi cách làm hoặc trả lời người dùng."
                else:
                    result = self._execute(c["name"], c["arguments"])
                    if seen[key] == 3:
                        result += "\n[Lưu ý: đây là lần thứ 3 bạn gọi y hệt. Hãy đổi cách tiếp cận.]"
                self.ui.tool_result(result)
                if self.memory and c["name"] not in ("recall", "remember", "todo"):
                    self.memory.add(f"{c['name']} {json.dumps(c['arguments'], ensure_ascii=False)}\n{result}",
                                    "tool", self.turn)
                if self.text_tools:  # không dùng tool role native: đưa kết quả như tin nhắn user
                    self.messages.append({"role": "user", "content": f"<tool_result>\n{result}\n</tool_result>"})
                else:
                    self.messages.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": result})
        else:
            self.ui.error(f"dừng: quá {self.max_steps} bước")
        if self.memory:
            self.memory.add(f"Hỏi: {user_input}\nĐáp: {final}", "turn", self.turn)
        self.ui.footer(time.time() - t0, steps)
        return final

"""Vòng lặp agent: model -> tool call -> kết quả -> model ... cho đến khi trả lời xong."""
from __future__ import annotations

import io
import json
import platform
import re
import time
from datetime import date
from typing import Callable

from .client import LLMError, LMStudioClient
from .memory import Memory, format_hits
from .tools import P, Tool, Workspace, clip
from .ui import UI

SYSTEM = """Bạn là trợ lý lập trình chạy trong terminal của người dùng.
Thư mục làm việc: thư mục hiện tại `.` ({name}) · Hệ điều hành: {os} · Hôm nay: {today}
Quy tắc:
- {effort_rule}
- Xem file thật (read_file/grep/glob) trước khi sửa; không đoán nội dung.
- Bạn bị nhốt trong thư mục workspace và bash chạy ngay tại đó: luôn dùng đường dẫn TƯƠNG ĐỐI (đường dẫn tuyệt đối có thể chứa dấu cách và dễ gõ sai), không cd ra ngoài, không đoán đường dẫn.
- Việc nhiều bước: ghi kế hoạch bằng tool todo rồi cập nhật khi xong từng bước.
- Sửa nhỏ, chính xác bằng edit_file thay vì ghi đè cả file.
- Sau khi sửa, chạy lệnh kiểm tra nếu có thể. Không lặp lại cùng một lệnh khi đã có kết quả.
- Trả lời ngắn gọn, cùng ngôn ngữ với người dùng, dùng markdown tối thiểu."""

EFFORT_RULES = {
    "low": "Suy nghĩ ngắn gọn. Câu hỏi đơn giản thì trả lời thẳng, không cần tool.",
    "medium": "Câu hỏi đơn giản thì trả lời thẳng; việc phức tạp thì suy nghĩ trước khi làm.",
    "high": ("Suy nghĩ kỹ trước khi hành động: làm rõ yêu cầu, cân nhắc ít nhất hai phương án, "
             "dự đoán rủi ro/trường hợp biên, và sau khi làm xong hãy tự kiểm tra lại kết quả."),
}

MEMORY_PROMPT = """
- Bạn có bộ nhớ ngoài. Hội thoại cũ và kết quả tool cũ được cất ở đó và KHÔNG còn trong context. Khi cần chi tiết đã nói/làm trước đó, gọi recall; việc đáng nhớ lâu dài thì gọi remember. Mục "Ký ức liên quan" (nếu có) là thứ hệ thống tự tìm giúp bạn, chỉ dùng khi hữu ích."""

def fingerprint(out: str):
    """Dấu vân của kết quả kiểm tra: tập tên test thất bại (unittest/pytest-style) hoặc đoạn cuối của output."""
    names = re.findall(r"^(?:FAIL|ERROR): (.+?)\s*$", out, re.M)
    return frozenset(names) if names else out.strip()[-400:]


def _no_regression(now, baseline) -> bool:
    """Lỗi hiện tại có nằm trọn trong các lỗi đã biết từ trước không (tức không có hồi quy mới)."""
    if baseline is None:
        return False
    if isinstance(now, frozenset) and isinstance(baseline, frozenset):
        return now <= baseline
    return now == baseline


REPEAT_FAIL_ADVICE = ("\n[Gợi ý] Kết quả kiểm tra GIỐNG HỆT lần trước dù bạn đã sửa: giả thuyết hiện tại có thể sai. "
                      "Hãy xét chính TEST có sai không (ngày giờ cố định cứng phụ thuộc hôm nay, trạng thái dùng chung giữa các "
                      "test, thứ tự chạy, kỳ vọng sai) thay vì tiếp tục sửa mã; đối chiếu dữ kiện cụ thể với thông báo lỗi.")

SUMMARY_SYSTEM = ("Bạn tóm tắt hội thoại giữa người dùng và trợ lý lập trình. Viết tiếng Việt, tối đa 120 từ, "
                  "dạng gạch đầu dòng. Giữ: yêu cầu của người dùng, quyết định, file đã tạo/sửa, giá trị và định danh "
                  "quan trọng (tên, mã, đường dẫn). Bỏ chi tiết vụn vặt. Chỉ trả về bản tóm tắt.")

SUBAGENT_RULES = ("Bạn là subagent khảo sát, chỉ đọc. Hoàn thành việc được giao rồi trả về kết luận ngắn gọn "
                  "(dưới 200 từ), kèm đường dẫn file:dòng làm bằng chứng. Không hỏi lại.")

SUBAGENT_TOOLS = ("read_file", "list_dir", "glob", "grep", "web_fetch", "web_search")

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
                 auto_approve: bool = True, text_tools: bool = False, max_steps: int = 25,
                 approve: Callable[[str, dict], bool] | None = None,
                 system_extra: str = "", max_context_chars: int = 40000, max_context_tokens: int | None = None,
                 ui: UI | None = None, memory: Memory | None = None, effort: str = "medium",
                 summarize: bool = False, ctx_total: int | None = None,
                 verify: Callable[[], tuple[bool, str]] | None = None, verify_rounds: int = 3):
        self.client, self.tools, self.ws = client, tools, ws
        self.auto_approve, self.text_tools, self.max_steps = auto_approve, text_tools, max_steps
        self.approve = approve or self._ask
        self.ui = ui or UI()
        self.system_extra, self.max_context_chars, self.max_context_tokens = system_extra, max_context_chars, max_context_tokens
        self.memory, self.effort, self.summarize, self.ctx_total = memory, effort, summarize, ctx_total
        self.verify, self.verify_rounds = verify, verify_rounds
        self.exhausted = False
        self.verify_baseline = None   # lỗi đã biết từ trước (chế độ --goal): không tính là hồi quy
        self.turn, self.first_live_turn, self._recalled = 0, 1, ""
        self.summary = ""
        self.ledger: list[str] = []   # nhật ký hành động (ghi/sửa/xoá/bash) của các bước đã rời khỏi context
        self._cpt, self._shrink, self._schema_len, self.last_prompt_tokens = 3.0, 1.0, 0, 0
        self._base_system = ""
        self.messages: list[dict] = []
        self.reset()

    # ---------- system prompt ----------
    def reset(self):
        system = SYSTEM.format(effort_rule=EFFORT_RULES[self.effort], name=self.ws.root.name, os=platform.system(), today=date.today().isoformat())
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
        self._base_system, self.summary, self.ledger = system, "", []
        self.messages = [{"role": "system", "content": system}]

    def _apply_system(self):
        extra = f"\n\nTóm tắt hội thoại trước đó (đã rời khỏi context):\n{self.summary}" if self.summary else ""
        if self.ledger:
            extra += ("\n\nNHẬT KÝ VIỆC BẠN ĐÃ LÀM (hệ thống tự ghi, gồm cả các bước đã rời khỏi context; ĐỪNG làm lại, "
                      "hãy đọc file để xem trạng thái hiện tại):\n" + "\n".join(f"- {x}" for x in self.ledger[-40:]))
        self.messages[0] = {"role": "system", "content": self._base_system + extra}

    ACTION_TOOLS = ("write_file", "edit_file", "move_file", "delete_file", "bash")

    def _ledger_add(self, evicted: list[dict]) -> None:
        """Ghi lại, một cách xác định (không nhờ model), những hành động có tác dụng phụ của các bước sắp bị loại."""
        for m in evicted:
            for tc in m.get("tool_calls") or []:
                f = tc.get("function", {})
                name = f.get("name", "")
                if name not in self.ACTION_TOOLS:
                    continue
                try:
                    a = json.loads(f.get("arguments") or "{}")
                except json.JSONDecodeError:
                    a = {}
                detail = a.get("command") if name == "bash" else a.get("path") or a.get("src") or ""
                if name == "write_file":
                    detail = f"{detail} ({len(str(a.get('content', '')).splitlines())} dòng)"
                line = clip(f"{name}: {str(detail).strip()}", 110).replace("\n", " ⏎ ")
                if self.ledger and self.ledger[-1].split(" (×")[0] == line:
                    n = int(self.ledger[-1].rsplit("(×", 1)[1].rstrip(")")) + 1 if "(×" in self.ledger[-1] else 2
                    self.ledger[-1] = f"{line} (×{n})"
                else:
                    self.ledger.append(line)
        self.ledger = self.ledger[-80:]

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

    # ---------- quản lý context ----------
    @staticmethod
    def _msg_len(m: dict) -> int:
        """Độ dài một tin nhắn, tính cả tham số tool_calls (vd nội dung file trong write_file chiếm rất nhiều token)."""
        n = len(m.get("content") or "")
        for tc in m.get("tool_calls") or []:
            f = tc.get("function", {})
            n += len(f.get("name", "")) + len(f.get("arguments") or "")
        return n

    def _size(self) -> int:
        n = sum(self._msg_len(m) for m in self.messages)
        return n + (self._schema_len if self.max_context_tokens else 0)

    def _budget(self) -> int:
        base = self.max_context_tokens * self._cpt if self.max_context_tokens else self.max_context_chars
        return int(base * self._shrink)

    def ctx_info(self) -> str:
        if not self.last_prompt_tokens:
            return ""
        k = lambda n: f"{n / 1000:.1f}k" if n >= 1000 else str(n)
        return f"{k(self.last_prompt_tokens)}/{k(self.ctx_total)}" if self.ctx_total else k(self.last_prompt_tokens)

    def _calibrate(self, view: list[dict], prompt_tokens: int) -> None:
        """Học tỉ lệ ký tự/token thật từ server, để ngân sách tính bằng token chính xác dần."""
        self.last_prompt_tokens = prompt_tokens
        chars = sum(self._msg_len(m) for m in view) + self._schema_len
        if prompt_tokens > 50 and chars:
            self._cpt = min(6.0, max(1.5, 0.5 * self._cpt + 0.5 * chars / prompt_tokens))

    def compact(self) -> int:
        """Giữ context dưới ngân sách. B1: lược kết quả tool cũ. B2: đẩy cả lượt cũ ra khỏi context
        (đã nằm trong bộ nhớ ngoài; nếu bật summarize thì gộp vào bản tóm tắt). Luôn giữ 2 lượt gần nhất."""
        budget = self._budget()
        if self._size() <= budget:
            return 0
        # Dọn một lần cho tụt xuống 70% ngân sách rồi để nó tự lớn lên: dọn từng chút mỗi bước sẽ đổi phần đầu của prompt,
        # làm server mất cache tiền tố và phải xử lý lại toàn bộ context (chậm gấp nhiều lần), còn tốn thêm lượt tóm tắt.
        n, budget = 0, int(budget * 0.7)
        for m in self.messages[1:-6]:
            if self._size() <= budget:
                break
            c = m.get("content") or ""
            is_result = m["role"] == "tool" or c.startswith("<tool_result>")
            if is_result and len(c) > 200 and not c.startswith("[đã lược"):
                m["content"] = f"[đã lược {len(c)} ký tự kết quả cũ" + (" — dùng recall để xem lại]" if self.memory else "]")
                n += 1
        for m in self.messages[1:-6]:
            if self._size() <= budget:
                break
            if m["role"] == "assistant":
                n += self._shrink_tool_args(m)
        if self.memory or self.summarize:
            evicted: list[dict] = []
            # B2: đẩy cả lượt cũ ra, cho đến khi vừa ngân sách (tối thiểu giữ lượt hiện tại)
            while self._size() > budget:
                starts = [i for i, m in enumerate(self.messages) if "_turn" in m]
                if len(starts) <= 1:
                    break
                evicted += self.messages[starts[0]:starts[1]]
                del self.messages[starts[0]:starts[1]]
                self.first_live_turn = self.messages[starts[0]]["_turn"]
                n += 1
            # B3: một lượt quá dài (agent chạy hàng chục bước): bỏ các bước cũ của chính lượt đó
            while self._size() > budget:
                span = self._oldest_step_group()
                if not span:
                    break
                evicted += self.messages[span[0]:span[1]]
                del self.messages[span[0]:span[1]]
                n += 1
            if evicted:
                self._ledger_add(evicted)
                self._apply_system()
            if evicted and self.summarize:
                self._update_summary(evicted)
        return n

    @staticmethod
    def _shrink_tool_args(m: dict, limit: int = 300) -> int:
        """Thay các chuỗi dài trong arguments của tool_calls cũ bằng dấu vết ngắn (vẫn là JSON hợp lệ)."""
        changed = 0
        for tc in m.get("tool_calls") or []:
            f = tc.get("function", {})
            raw = f.get("arguments") or ""
            if len(raw) <= limit:
                continue
            try:
                args = json.loads(raw)
            except json.JSONDecodeError:
                continue
            short = {k: (f"[đã lược {len(v)} ký tự, nội dung đã nằm trên đĩa/bộ nhớ]" if isinstance(v, str) and len(v) > limit else v)
                     for k, v in args.items()}
            f["arguments"] = json.dumps(short, ensure_ascii=False)
            changed += 1
        return changed

    def _oldest_step_group(self) -> tuple[int, int] | None:
        """Nhóm bước cũ nhất (assistant + các kết quả tool của nó) trong lượt hiện tại, hoặc None nếu còn quá ít
        để dọn. Luôn giữ nguyên tin nhắn người dùng của lượt và ít nhất 6 tin gần nhất (không làm lệch cặp tool_call/tool)."""
        users = [i for i, m in enumerate(self.messages) if "_turn" in m]
        if not users:
            return None
        base = users[-1] + 1
        rest = self.messages[base:]
        end = next((i for i, m in enumerate(rest) if i > 0 and m["role"] == "assistant"), None)
        if end is None or len(rest) - end < 6:
            return None
        return base, base + end

    @staticmethod
    def _transcript(msgs: list[dict]) -> str:
        lines = []
        for m in msgs:
            c = (m.get("content") or "").strip()
            if m["role"] == "user" and "_turn" in m:
                lines.append(f"Người dùng: {clip(c, 600)}")
            elif m["role"] == "assistant":
                tools = ", ".join(t["function"]["name"] for t in m.get("tool_calls", []))
                lines.append(f"Trợ lý: {clip(c, 600)}" + (f" [gọi tool: {tools}]" if tools else ""))
            elif m["role"] == "tool" or c.startswith("<tool_result>"):
                lines.append(f"Kết quả tool: {clip(c, 300)}")
        return clip("\n".join(lines), 6000)

    def _update_summary(self, evicted: list[dict]) -> None:
        transcript = self._transcript(evicted)
        if not transcript:
            return
        prompt = [{"role": "system", "content": SUMMARY_SYSTEM},
                  {"role": "user", "content": f"Tóm tắt hiện có:\n{self.summary or '(chưa có)'}\n\n"
                                              f"Hội thoại mới cần gộp vào:\n{transcript}"}]
        try:
            text = self.client.chat(prompt, None).content.strip()
        except Exception:
            return  # tóm tắt lỗi thì thôi: nội dung vẫn còn trong bộ nhớ ngoài
        if text:
            self.summary = clip(text, 1500)
            self._apply_system()
            self.ui.summary_note()

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

    def drop_dangling_turn(self) -> bool:
        """Lượt vừa lỗi trước khi model kịp làm gì: bỏ tin nhắn người dùng đang treo để gõ lại không bị trùng lượt."""
        last = self.messages[-1] if self.messages else {}
        if last.get("role") == "user" and last.get("_turn") == self.turn:
            self.messages.pop()
            self.turn -= 1
            return True
        return False

    # ---------- lưu / mở lại phiên ----------
    def save(self, path) -> None:
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {"messages": self.messages, "summary": self.summary, "ledger": self.ledger, "turn": self.turn,
                "first_live_turn": self.first_live_turn,
                "memory_session": self.memory.session if self.memory else None}
        Path(path).write_text(json.dumps(data, ensure_ascii=False))

    def load(self, path) -> bool:
        from pathlib import Path
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            return False
        self.messages, self.summary = data["messages"], data.get("summary", "")
        self.ledger = data.get("ledger", [])
        self.turn, self.first_live_turn = data["turn"], data["first_live_turn"]
        if self.memory and data.get("memory_session"):
            self.memory.session = data["memory_session"]
        self._apply_system()  # lấy prompt hệ thống mới nhất + tóm tắt đã lưu
        return True

    # ---------- nhắc nhở ----------
    def _nudge(self, hist: list[tuple[str, dict]], steps: int, st: dict) -> str:
        """Nhắc agent khi thấy kiểu làm việc kém hiệu quả (tối đa một lần mỗi 12 bước)."""
        if steps - st["nudged"] < 12:
            return ""
        if "todo" in self.tools and steps - st["todo"] >= 20:
            st["nudged"] = steps
            return (f"[Nhắc] Đã {steps - st['todo']} bước mà bạn chưa cập nhật kế hoạch. Hãy gọi tool todo để ghi việc đã "
                    "xong/đang làm/còn lại (theo thứ tự ưu tiên của yêu cầu), rồi tiếp tục.")
        last = hist[-8:]
        if len(last) == 8 and sum(1 for n, a in last if n == "bash" and "python3 -c" in str(a.get("command", ""))) >= 6:
            st["nudged"] = steps
            return ("[Nhắc] Bạn đang chạy script python -c lẻ nhiều lần liên tiếp. Hãy dừng mò: dùng read_file đọc đúng đoạn "
                    "mã liên quan (có số dòng) hoặc grep, nêu giả thuyết về nguyên nhân, rồi sửa bằng edit_file và chạy lại test.")
        if steps - st.get("edit", 0) >= 25:
            st["nudged"] = steps
            return (f"[Nhắc] Đã {steps - st.get('edit', 0)} bước mà bạn chưa sửa được file nào: có thể bạn đang mò hoặc lặp. "
                    "Hãy đổi chiến lược: đọc lại thông báo lỗi thật (read_file/grep), chia nhỏ vấn đề, chọn MỘT giả thuyết, "
                    "sửa bằng edit_file rồi chạy lại test; nếu kẹt ở một chỗ quá lâu thì chuyển sang việc ưu tiên kế tiếp "
                    "và ghi chú chỗ kẹt trong todo.")
        return ""

    # ---------- vòng lặp ----------
    _OVERFLOW = re.compile(r"context|exceed|too (long|large)|n_ctx|n_keep|prompt.*tokens", re.I)

    NET_RETRY_DELAY = 2.0   # giây nghỉ trước khi thử lại sau lỗi đường truyền

    def _chat(self, schemas):
        overflow_retried = net_retried = False
        while True:
            self._schema_len = len(json.dumps(schemas)) if schemas else 0
            self.ui.spinner_start()
            try:
                return self.client.chat(self._view(), schemas,
                                        on_token=self.ui.stream_answer, on_thinking=self.ui.stream_think)
            except LLMError as e:
                if not overflow_retried and self._OVERFLOW.search(str(e)):
                    overflow_retried = True
                    self._shrink *= 0.6  # tràn context: siết ngân sách rồi thử lại một lần
                    self.compact()
                    continue
                if getattr(e, "transient", False) and not net_retried:
                    net_retried = True   # đứt kết nối/luồng hỏng: nghỉ một chút rồi thử lại đúng một lần
                    self.ui.error(f"lỗi đường truyền tới server model ({str(e)[:100]}), thử lại sau {self.NET_RETRY_DELAY:g}s")
                    time.sleep(self.NET_RETRY_DELAY)
                    continue
                raise
            finally:
                self.ui.end_stream()

    def run(self, user_input: str) -> str:
        self.turn += 1
        self._recalled, self._shrink = "", 1.0
        self.exhausted = False   # True nếu lượt này dừng vì hết ngân sách bước (không phải kết thúc tự nhiên)
        self.messages.append({"role": "user", "content": user_input, "_turn": self.turn})
        schemas = None if self.text_tools else [t.schema() for t in self.tools.values()]
        self._schema_len = len(json.dumps(schemas)) if schemas else 0
        if self.memory:
            self.compact()  # dọn trước, để first_live_turn phản ánh đúng những gì còn trong context
            hits = self.memory.search(user_input, k=4, first_live_turn=self.first_live_turn)
            self._recalled = format_hits(hits)
            if hits:
                self.ui.memory_note(len(hits))
        final, t0, steps, seen, rounds, last_ok, prev_fp = "", time.time(), 0, {}, 0, None, None
        empties = 0
        hist: list[tuple[str, dict]] = []
        nst = {"nudged": 0, "todo": 0}
        for _ in range(self.max_steps):
            steps += 1
            self.compact()
            reply = self._chat(schemas)
            if reply.prompt_tokens:
                self._calibrate(self._view(), reply.prompt_tokens)
            if reply.aborted and not reply.tool_calls:
                why = ("bị lặp một câu nhiều lần" if reply.aborted == "repetition" else "quá dài mà chưa hành động")
                self.ui.error(f"suy nghĩ của model {why}, đã ngắt luồng")
                self.messages.append({"role": "assistant", "content": reply.content or "(suy nghĩ bị ngắt)"})
                self.messages.append({"role": "user", "content": (
                    f"[Hệ thống] Suy nghĩ của bạn {why} nên bị ngắt. Đừng phân tích vòng tròn nữa: chọn MỘT hành động cụ thể "
                    "(đọc đúng đoạn file bằng read_file, sửa bằng edit_file, hoặc chạy test) và làm ngay; nếu chưa chắc nguyên "
                    "nhân thì đọc mã/thông báo lỗi thật để lấy thêm dữ kiện.")})
                continue
            calls = reply.tool_calls or parse_text_tool_calls(reply.content)
            if not calls and not reply.content.strip() and empties < 2:
                # model chỉ "nghĩ" rồi trả về rỗng (không nói gì, không gọi tool): đó không phải là kết thúc
                empties += 1
                self.ui.error(f"model trả lời rỗng (chỉ suy nghĩ), nhắc và thử lại ({empties}/2)")
                self.messages.append({"role": "assistant", "content": "(không có nội dung)"})
                self.messages.append({"role": "user", "content": (
                    "[Hệ thống] Bạn chưa trả lời gì và chưa gọi tool nào. Hãy thực hiện ngay hành động bạn vừa định làm "
                    "bằng cách GỌI TOOL, hoặc nếu đã xong thì báo kết quả bằng câu trả lời có nội dung.")})
                continue
            msg: dict = {"role": "assistant", "content": reply.content}
            if reply.tool_calls:
                msg["tool_calls"] = [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                    for c in reply.tool_calls]
            self.messages.append(msg)
            if not calls:
                final = reply.content
                if self.verify and rounds < self.verify_rounds:
                    last_ok, out = self.verify()
                    fp = fingerprint(out)
                    if not last_ok and _no_regression(fp, self.verify_baseline):
                        self.ui.w(self.ui.c("muted", "◌ kiểm tra: chỉ còn lỗi đã có từ trước, không có hồi quy mới → tính là đạt"))
                        last_ok = True
                    else:
                        self.ui.verify_note(last_ok, rounds + 1, self.verify_rounds)
                    if not last_ok:
                        rounds += 1
                        advice = REPEAT_FAIL_ADVICE if fp == prev_fp else ""
                        prev_fp = fp
                        self.messages.append({"role": "user", "content": (
                            "[Kiểm tra tự động] Bạn báo xong nhưng lệnh kiểm tra THẤT BẠI, nên bạn CHƯA xong:\n"
                            f"{clip(out, 3000)}{advice}\nHãy tìm nguyên nhân thật rồi sửa, chạy lại kiểm tra và chỉ báo xong khi nó đạt.")})
                        continue
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
                hist.append((c["name"], c["arguments"]))
                if c["name"] == "todo":
                    nst["todo"] = steps
                if c["name"] in ("edit_file", "write_file") and not result.startswith("LỖI"):
                    nst["edit"] = steps
                if note := self._nudge(hist, steps, nst):
                    result += "\n" + note
                self.ui.tool_result(result)
                if self.memory and c["name"] not in ("recall", "remember", "todo"):
                    self.memory.add(f"{c['name']} {json.dumps(c['arguments'], ensure_ascii=False)}\n{result}",
                                    "tool", self.turn)
                if self.text_tools:  # không dùng tool role native: đưa kết quả như tin nhắn user
                    self.messages.append({"role": "user", "content": f"<tool_result>\n{result}\n</tool_result>"})
                else:
                    self.messages.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": result})
        else:
            self.exhausted = True
            self.ui.error(f"dừng: quá {self.max_steps} bước")
        if self.verify and last_ok is False:
            self.ui.error(f"kiểm tra tự động VẪN thất bại sau {self.verify_rounds} vòng: kết quả chưa đáng tin")
        if self.memory:
            self.memory.add(f"Hỏi: {user_input}\nĐáp: {final}", "turn", self.turn)
        self.ui.footer(time.time() - t0, steps, self.ctx_info())
        return final


def delegate_tool(parent: Agent) -> Tool:
    """Tool `delegate`: giao việc khảo sát chỉ-đọc cho subagent có context riêng, chỉ nhận lại kết luận."""
    def delegate(task: str) -> str:
        sub = Agent(parent.client, {n: t for n, t in parent.tools.items() if n in SUBAGENT_TOOLS}, parent.ws,
                    ui=UI(io.StringIO()), memory=None, effort=parent.effort, max_steps=12,
                    auto_approve=parent.auto_approve, approve=parent.approve,   # --ask áp dụng cả cho subagent (web_fetch/web_search)
                    system_extra=SUBAGENT_RULES, max_context_chars=parent.max_context_chars,
                    max_context_tokens=parent.max_context_tokens, text_tools=parent.text_tools)
        return clip(sub.run(task).strip() or "(subagent không trả về kết quả)", 3000)

    return Tool("delegate", "Giao một việc khảo sát chỉ-đọc (đọc nhiều file, tìm hiểu mã, tra cứu web) cho subagent "
                "chạy trong context riêng và chỉ nhận lại kết luận. Dùng khi không muốn làm đầy context của bạn.",
                {"task": P("string", "mô tả đầy đủ việc cần khảo sát và câu hỏi cần trả lời")}, delegate, read_only=True)

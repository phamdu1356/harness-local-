"""Client cho server OpenAI-compatible (LM Studio: http://localhost:1234/v1), stdlib, streaming."""
from __future__ import annotations

import http.client
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Reply:
    content: str = ""
    thinking: str = ""
    tool_calls: list[dict] = field(default_factory=list)  # [{"id", "name", "arguments"(dict)}]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    aborted: str = ""   # "repetition" | "too_long": luồng bị ngắt giữa chừng vì model lặp/lan man


class LLMError(RuntimeError):
    """Lỗi khi nói chuyện với server model. transient=True: lỗi đường truyền (đứt/quá hạn/luồng hỏng) đáng thử lại một lần."""

    def __init__(self, msg: str = "", transient: bool = False):
        super().__init__(msg)
        self.transient = transient


MAX_THINK_CHARS = 14000
_SEG_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_EMBED_RX = re.compile(r"embed|(?:bge|e5|gte)[-_]|minilm|nomic-embed|arctic-embed", re.I)


def detect_repetition(text: str, repeats: int = 3) -> bool:
    """Một khối 1–4 câu (tổng ≥20 ký tự) lặp liên tiếp `repeats` lần ở cuối văn bản: vòng lặp suy nghĩ của model."""
    segs = [x.strip() for x in _SEG_SPLIT.split(text[-3000:]) if x.strip()]
    for p in range(1, 5):
        if len(segs) < p * repeats:
            break
        block = segs[-p:]
        if sum(len(x) for x in block) < 20:
            continue
        if all(segs[len(segs) - (k + 1) * p: len(segs) - k * p] == block for k in range(1, repeats)):
            return True
    return False


class LMStudioClient:
    def __init__(self, model: str | None = None, host: str = "http://localhost:1234",
                 temperature: float = 0.2, timeout: int = 600):
        self.model, self.host = model, host.rstrip("/")
        self.temperature, self.timeout = temperature, timeout

    def list_models(self) -> list[str]:
        try:
            with urllib.request.urlopen(f"{self.host}/v1/models", timeout=5) as r:
                data = json.load(r)
            if not isinstance(data, dict) or not isinstance(data.get("data", []), list):
                raise LLMError(f"Server {self.host}/v1/models trả dữ liệu không hợp lệ.")
            return [m["id"] for m in data.get("data", []) if isinstance(m, dict) and isinstance(m.get("id"), str)]
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode(errors="replace")
            finally:
                e.close()
            raise LLMError(f"HTTP {e.code} từ {self.host}/v1/models: {body}") from e
        except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
            raise LLMError(f"Không kết nối được LM Studio tại {self.host}. "
                           "Mở LM Studio → tab Developer → Start Server.") from e
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError) as e:
            raise LLMError(f"Server {self.host}/v1/models trả JSON hỏng: {e}") from e

    def context_length(self) -> int | None:
        """Context đang nạp của model: LM Studio (/api/v0/models) hoặc llama-server (/props)."""
        for path, pick in (
            ("/api/v0/models", lambda d: next((m.get("loaded_context_length") for m in d.get("data", [])
                                               if m.get("id") == self.model and m.get("state") == "loaded"), None)),
            ("/props", lambda d: d.get("default_generation_settings", {}).get("n_ctx")),
        ):
            try:
                with urllib.request.urlopen(f"{self.host}{path}", timeout=3) as r:
                    if n := pick(json.load(r)):
                        return int(n)
            except urllib.error.HTTPError as e:
                e.close()
                continue
            except Exception:
                continue
        return None

    def ensure_model(self) -> str:
        if not self.model:  # dùng model đầu tiên đang load trong LM Studio
            models = [m for m in self.list_models() if not _EMBED_RX.search(m)]
            if not models:
                raise LLMError("LM Studio chưa có model nào được load.")
            self.model = models[0]
        return self.model

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             on_token: Callable[[str], None] | None = None,
             on_thinking: Callable[[str], None] | None = None) -> Reply:
        body = {"model": self.ensure_model(), "messages": messages, "stream": True,
                "stream_options": {"include_usage": True}, "temperature": self.temperature}
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(
            f"{self.host}/v1/chat/completions", data=json.dumps(body, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json"})
        reply, partial, checked = Reply(), {}, 0  # partial: index -> {id, name, args(str)}
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw in resp:
                    line = raw.decode(errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError as e:
                        raise LLMError(f"Server trả dòng stream hỏng: {data[:80]!r}", transient=True) from e
                    if not isinstance(chunk, dict):
                        continue
                    if chunk.get("error"):
                        raise LLMError(str(chunk["error"]))
                    u = chunk.get("usage")
                    if isinstance(u, dict):
                        reply.prompt_tokens = u.get("prompt_tokens") or 0
                        reply.completion_tokens = u.get("completion_tokens") or 0
                    for choice in chunk.get("choices") or []:
                        if not isinstance(choice, dict):
                            continue
                        d = choice.get("delta") or {}
                        if not isinstance(d, dict):
                            continue
                        if t := d.get("reasoning_content") or d.get("reasoning"):
                            reply.thinking += t
                            if on_thinking:
                                on_thinking(t)
                        if c := d.get("content"):
                            reply.content += c
                            if on_token:
                                on_token(c)
                        for tc in d.get("tool_calls") or []:
                            if not isinstance(tc, dict):
                                continue
                            idx = tc.get("index")
                            if not isinstance(idx, int) or isinstance(idx, bool):
                                idx = 0
                            p = partial.setdefault(idx, {"id": "", "name": "", "args": ""})
                            p["id"] = tc.get("id") or p["id"]
                            fn = tc.get("function") or {}
                            if not isinstance(fn, dict):
                                fn = {}
                            p["name"] += fn.get("name") or ""
                            raw_args = fn.get("arguments")
                            if isinstance(raw_args, dict):
                                p["args"] += json.dumps(raw_args, ensure_ascii=False)
                            elif isinstance(raw_args, str):
                                p["args"] += raw_args
                    seen = len(reply.thinking) + len(reply.content)
                    if seen - checked >= 200:     # kiểm tra mỗi ~200 ký tự mới, không phải từng chunk
                        checked = seen
                        if detect_repetition(reply.thinking) or detect_repetition(reply.content):
                            reply.aborted = "repetition"
                        elif len(reply.thinking) > MAX_THINK_CHARS and not reply.content and not partial:
                            reply.aborted = "too_long"
                    if reply.aborted:
                        break
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode(errors="replace")
            finally:
                e.close()
            raise LLMError(f"HTTP {e.code}: {body}") from e
        except urllib.error.URLError as e:
            raise LLMError(f"Mất kết nối LM Studio tại {self.host}: {e.reason}", transient=True) from e
        except (OSError, http.client.HTTPException) as e:   # quá hạn, bị ngắt giữa chừng khi đang stream (server chết/OOM)
            raise LLMError(f"Đứt kết nối với server model tại {self.host} khi đang nhận câu trả lời: "
                           f"{type(e).__name__}: {e}", transient=True) from e
        for i in sorted(partial):
            p = partial[i]
            raw_s = p["args"].strip()
            arg_error = ""
            if not raw_s:
                args = {}
            else:
                try:
                    parsed = json.loads(raw_s)
                    if parsed is None:
                        args = {}
                    elif isinstance(parsed, dict):
                        args = parsed
                    else:
                        args = {}
                        arg_error = f"tham số tool phải là object JSON, nhận {type(parsed).__name__}: {raw_s[:120]}"
                except json.JSONDecodeError as e:
                    args = {}
                    arg_error = f"JSON tham số tool hỏng ({e.msg}): {raw_s[:120]}"
            tc_item = {"id": p["id"] or f"call_{i}", "name": p["name"], "arguments": args}
            if arg_error:
                tc_item["arg_error"] = arg_error
                tc_item["raw_arguments"] = p["args"]
            reply.tool_calls.append(tc_item)
        return reply

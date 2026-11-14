"""Client cho server OpenAI-compatible (LM Studio: http://localhost:1234/v1), stdlib, streaming."""
from __future__ import annotations

import json
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


class LLMError(RuntimeError):
    pass


class LMStudioClient:
    def __init__(self, model: str | None = None, host: str = "http://localhost:1234",
                 temperature: float = 0.2, timeout: int = 600):
        self.model, self.host = model, host.rstrip("/")
        self.temperature, self.timeout = temperature, timeout

    def list_models(self) -> list[str]:
        try:
            with urllib.request.urlopen(f"{self.host}/v1/models", timeout=5) as r:
                return [m["id"] for m in json.load(r).get("data", [])]
        except (urllib.error.URLError, OSError) as e:
            raise LLMError(f"Không kết nối được LM Studio tại {self.host}. "
                           "Mở LM Studio → tab Developer → Start Server.") from e

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
            except Exception:
                continue
        return None

    def ensure_model(self) -> str:
        if not self.model:  # dùng model đầu tiên đang load trong LM Studio
            models = [m for m in self.list_models() if "embed" not in m]
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
            f"{self.host}/v1/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        reply, partial = Reply(), {}  # partial: index -> {id, name, args(str)}
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw in resp:
                    line = raw.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if "error" in chunk:
                        raise LLMError(str(chunk["error"]))
                    if u := chunk.get("usage"):
                        reply.prompt_tokens = u.get("prompt_tokens", 0)
                        reply.completion_tokens = u.get("completion_tokens", 0)
                    for choice in chunk.get("choices", []):
                        d = choice.get("delta", {})
                        if t := d.get("reasoning_content") or d.get("reasoning"):
                            reply.thinking += t
                            if on_thinking:
                                on_thinking(t)
                        if c := d.get("content"):
                            reply.content += c
                            if on_token:
                                on_token(c)
                        for tc in d.get("tool_calls") or []:
                            p = partial.setdefault(tc.get("index", 0), {"id": "", "name": "", "args": ""})
                            p["id"] = tc.get("id") or p["id"]
                            fn = tc.get("function", {})
                            p["name"] += fn.get("name") or ""
                            p["args"] += fn.get("arguments") or ""
        except urllib.error.HTTPError as e:
            raise LLMError(f"HTTP {e.code}: {e.read().decode(errors='replace')}") from e
        except urllib.error.URLError as e:
            raise LLMError(f"Mất kết nối LM Studio tại {self.host}: {e.reason}") from e
        for i in sorted(partial):
            p = partial[i]
            try:
                args = json.loads(p["args"]) if p["args"].strip() else {}
            except json.JSONDecodeError:
                args = {}
            reply.tool_calls.append({"id": p["id"] or f"call_{i}", "name": p["name"], "arguments": args})
        return reply

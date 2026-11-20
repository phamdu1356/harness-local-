"""Embedding qua llama-server --embeddings (chạy CPU). Vector lưu dạng float32, so cosine bằng Python thuần."""
from __future__ import annotations

import array
import json
import math
import urllib.request


class EmbedError(RuntimeError):
    pass


class EmbedClient:
    def __init__(self, host: str, model: str = "embed", timeout: int = 60):
        self.host, self.model, self.timeout = host.rstrip("/"), model, timeout

    def embed(self, texts: list[str]) -> list[array.array]:
        if not texts:
            return []
        body = json.dumps({"model": self.model, "input": texts}).encode()
        req = urllib.request.Request(f"{self.host}/v1/embeddings", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.load(r)["data"]
        except Exception as e:
            if hasattr(e, "close"):
                e.close()
            raise EmbedError(f"embedding lỗi: {e}") from e
        out = []
        for d in sorted(data, key=lambda x: x.get("index", 0)):
            v = array.array("f", d["embedding"])
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append(array.array("f", (x / norm for x in v)))
        return out


def cosine(a: array.array, b: array.array) -> float:
    """Vector đã chuẩn hoá nên cosine = tích vô hướng."""
    return sum(map(float.__mul__, a, b)) if len(a) == len(b) else -1.0

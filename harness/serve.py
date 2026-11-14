"""Tự chạy llama-server (llama.cpp) cho harness, không cần mở giao diện LM Studio.

Tìm binary: PATH, hoặc backend CUDA mới nhất của LM Studio. Tìm model: file .gguf trong ~/.lmstudio/models.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

LMS = Path.home() / ".lmstudio"


class ServeError(RuntimeError):
    pass


def _version(p: Path) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", p.parent.name.rsplit("-", 1)[-1]))


def find_server(cpu: bool = False) -> tuple[str, dict]:
    """Trả về (đường dẫn llama-server, biến môi trường cần thiết). cpu=True: backend CPU thuần, ẩn GPU."""
    if path := shutil.which("llama-server"):
        return path, {**os.environ, **({"CUDA_VISIBLE_DEVICES": ""} if cpu else {})}
    backends = LMS / "extensions" / "backends"
    cands = sorted(backends.glob("llama.cpp-*/llama-server"), key=_version)
    if cpu:
        cands = [c for c in cands if not re.search(r"cuda|vulkan", c.parent.name)]
    # ưu tiên CUDA (13 > 12 > cũ), rồi bản mới nhất; Vulkan/CPU là phương án cuối
    def rank(p: Path):
        n = p.parent.name
        tier = 3 if "cuda13" in n else 2 if "cuda12" in n else 1 if "cuda" in n else 0
        return (tier, _version(p))
    if not cands:
        raise ServeError("Không tìm thấy llama-server (PATH hoặc ~/.lmstudio/extensions/backends).")
    best = max(cands, key=rank)
    env = dict(os.environ)
    if cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""  # không chạm GPU/VRAM
    libs = [str(best.parent)]
    for v in ("cuda13", "cuda12"):
        if v in best.parent.name:
            libs.append(str(backends / "vendor" / f"linux-llama-{v}-vendor-v1"))
    if "cuda" in best.parent.name and "cuda1" not in best.parent.name:
        libs.append(str(backends / "vendor" / "linux-llama-cuda-vendor-v1"))
    env["LD_LIBRARY_PATH"] = ":".join(libs + [env.get("LD_LIBRARY_PATH", "")])
    return str(best), env


EMBED_RX = re.compile(r"embed|bge-|e5-|gte-|minilm|nomic-embed|arctic-embed", re.I)


def _ggufs(embedding: bool) -> list[Path]:
    return [p for p in (LMS / "models").rglob("*.gguf")
            if "mmproj" not in p.name.lower() and bool(EMBED_RX.search(p.name)) == embedding]


def find_embed_model(spec: str | None = None) -> Path:
    if spec and Path(spec).is_file():
        return Path(spec)
    c = [p for p in _ggufs(True) if not spec or spec.lower() in str(p).lower()]
    if not c:
        raise ServeError("Không tìm thấy model embedding .gguf trong ~/.lmstudio/models.")
    return min(c, key=lambda p: p.stat().st_size)


def find_model(spec: str | None) -> Path:
    if spec and Path(spec).is_file():
        return Path(spec)
    ggufs = _ggufs(False)  # model chat: loại model embedding
    if spec:
        ggufs = [p for p in ggufs if spec.lower() in str(p).lower()]
    if not ggufs:
        raise ServeError(f"Không tìm thấy model .gguf khớp {spec!r} trong {LMS / 'models'}.")
    return min(ggufs, key=lambda p: p.stat().st_size) if not spec else ggufs[0]


class LocalServer:
    def __init__(self, model: Path, port: int = 8081, ctx: int = 16384, gpu_layers: int = 99,
                 reasoning: str = "auto", kv_quant: str = "q8_0", cpu: bool = False, threads: int = 0):
        self.model, self.port, self.ctx = model, port, ctx
        self.cpu, self.threads = cpu, threads
        self.gpu_layers = 0 if cpu else gpu_layers
        self.reasoning, self.kv_quant = reasoning, kv_quant
        self.proc: subprocess.Popen | None = None
        self.log = Path(tempfile.gettempdir()) / "harness-llama-server.log"

    @property
    def host(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def command(self, binary: str) -> list[str]:
        return [binary, "-m", str(self.model), "--host", "127.0.0.1", "--port", str(self.port),
                "-c", str(self.ctx), "-ngl", str(self.gpu_layers), "-fa", "on",
                "-ctk", self.kv_quant, "-ctv", self.kv_quant, "--jinja", "--no-webui",
                "--reasoning", self.reasoning, "--alias", "local"] + (["-t", str(self.threads)] if self.threads else [])

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/health", timeout=1) as r:
                return r.status == 200
        except Exception:
            return False

    def start(self, timeout: int = 180) -> None:
        if self.healthy():
            return  # đã có server chạy sẵn trên cổng này
        binary, env = find_server(self.cpu)
        logf = open(self.log, "w")
        self.proc = subprocess.Popen(self.command(binary), env=env, stdout=logf, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                tail = self.log.read_text(errors="replace")[-800:]
                raise ServeError(f"llama-server thoát (mã {self.proc.returncode}). Log {self.log}:\n{tail}")
            if self.healthy():
                return
            time.sleep(0.5)
        self.stop()
        raise ServeError(f"llama-server không sẵn sàng sau {timeout}s. Xem {self.log}")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class EmbedServer(LocalServer):
    """llama-server chế độ embedding, luôn chạy CPU để không tranh VRAM với model chat."""

    def __init__(self, model: Path, port: int = 8082, threads: int = 4):
        super().__init__(model, port, ctx=8192, cpu=True, threads=threads)
        self.log = Path(tempfile.gettempdir()) / "harness-embed-server.log"

    def command(self, binary: str) -> list[str]:
        return [binary, "-m", str(self.model), "--host", "127.0.0.1", "--port", str(self.port),
                "-c", "8192", "-b", "2048", "-ub", "2048", "-ngl", "0", "--embeddings",
                "--no-webui", "--alias", "embed", "-t", str(self.threads)]

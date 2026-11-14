"""Client MCP (Model Context Protocol) qua stdio: nạp tool từ các MCP server do bạn cấu hình.

Cấu hình: <workspace>/.harness/mcp.json hoặc ~/.config/harness/mcp.json
  {"mcpServers": {"tên": {"command": "npx", "args": ["-y", "@pkg/server"], "env": {"KEY": "..."}}}}
"""
from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
from pathlib import Path

from .tools import Tool, clip

PROTOCOL = "2024-11-05"


class MCPError(RuntimeError):
    pass


class MCPServer:
    def __init__(self, name: str, command: str, args: list[str] | None = None, env: dict | None = None,
                 cwd: str | None = None, timeout: float = 30.0):
        self.name, self.timeout = name, timeout
        self.cmd = [command, *(args or [])]
        self.env, self.cwd = {**os.environ, **(env or {})}, cwd
        self.proc: subprocess.Popen | None = None
        self._q: queue.Queue = queue.Queue()
        self._id = 0

    def start(self) -> None:
        try:
            self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, text=True, bufsize=1, env=self.env, cwd=self.cwd)
        except OSError as e:
            raise MCPError(f"không chạy được {self.cmd[0]}: {e}") from e
        threading.Thread(target=self._reader, daemon=True).start()
        self.request("initialize", {"protocolVersion": PROTOCOL, "capabilities": {},
                                    "clientInfo": {"name": "harness", "version": "0.1"}})
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _reader(self) -> None:
        for line in self.proc.stdout:
            try:
                self._q.put(json.loads(line))
            except json.JSONDecodeError:
                continue
        self._q.put(None)  # server đóng stdout

    def _send(self, obj: dict) -> None:
        try:
            self.proc.stdin.write(json.dumps(obj) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise MCPError(f"MCP server {self.name} đã thoát") from e

    def request(self, method: str, params: dict | None = None):
        self._id += 1
        rid = self._id
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        while True:
            try:
                msg = self._q.get(timeout=self.timeout)
            except queue.Empty:
                raise MCPError(f"{self.name}: quá {self.timeout}s chờ {method}")
            if msg is None:
                raise MCPError(f"MCP server {self.name} đã thoát")
            if msg.get("id") == rid:
                if "error" in msg:
                    raise MCPError(f"{self.name}: {msg['error'].get('message', msg['error'])}")
                return msg.get("result", {})
            if "id" in msg and "method" in msg:  # server hỏi ngược (ping...): trả lời rỗng cho khỏi treo
                self._send({"jsonrpc": "2.0", "id": msg["id"], "result": {}})

    def list_tools(self) -> list[dict]:
        tools, cursor = [], None
        while True:
            res = self.request("tools/list", {"cursor": cursor} if cursor else {})
            tools += res.get("tools", [])
            cursor = res.get("nextCursor")
            if not cursor:
                return tools

    def call(self, tool: str, args: dict) -> str:
        res = self.request("tools/call", {"name": tool, "arguments": args})
        parts = [c.get("text", f"[{c.get('type', 'dữ liệu')}]") for c in res.get("content", [])]
        text = "\n".join(parts).strip() or json.dumps(res.get("structuredContent", ""), ensure_ascii=False)
        return clip(("LỖI: " if res.get("isError") else "") + text)

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def _wrap(server: MCPServer, spec: dict) -> Tool:
    schema = spec.get("inputSchema") or {}
    required = set(schema.get("required", []))
    params = {}
    for k, v in (schema.get("properties") or {}).items():
        v = dict(v) if isinstance(v, dict) else {}
        if not isinstance(v.get("type"), str):
            v["type"] = "string"
        params[k] = {**v, "optional": k not in required}
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", f"{server.name}__{spec['name']}")[:64]
    desc = f"[MCP {server.name}] {spec.get('description', spec['name'])}"
    return Tool(name, desc, params, lambda **kw: server.call(spec["name"], kw), needs_approval=True)


def load_config(ws_root: Path) -> dict:
    for p in (ws_root / ".harness" / "mcp.json", Path.home() / ".config" / "harness" / "mcp.json"):
        if p.is_file():
            data = json.loads(p.read_text())
            return data.get("mcpServers") or data.get("servers") or {}
    return {}


def load_mcp_tools(ws_root: Path, warn=print) -> tuple[dict[str, Tool], list[MCPServer]]:
    tools: dict[str, Tool] = {}
    servers: list[MCPServer] = []
    try:
        cfg = load_config(ws_root)
    except (OSError, json.JSONDecodeError) as e:
        warn(f"⚠ mcp.json lỗi: {e}")
        return tools, servers
    for name, c in cfg.items():
        srv = MCPServer(name, c["command"], c.get("args"), c.get("env"), str(ws_root))
        try:
            srv.start()
            for spec in srv.list_tools():
                t = _wrap(srv, spec)
                tools[t.name] = t
            servers.append(srv)
        except (MCPError, KeyError) as e:
            warn(f"⚠ MCP {name}: {e}")
            srv.stop()
    return tools, servers

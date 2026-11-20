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
        try:
            self._reader_thread = threading.Thread(target=self._reader, daemon=True)
            self._reader_thread.start()
            self.request("initialize", {"protocolVersion": PROTOCOL, "capabilities": {},
                                        "clientInfo": {"name": "harness", "version": "0.1"}})
            self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        except Exception:
            self.stop()
            raise

    def _reader(self) -> None:
        proc = self.proc
        if not proc or not proc.stdout:
            self._q.put(None)
            return
        try:
            for line in proc.stdout:
                try:
                    self._q.put(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except (ValueError, OSError):
            pass
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
        proc = self.proc
        if not proc:
            return
        self.proc = None
        try:
            if proc.stdin:
                try:
                    proc.stdin.close()
                except OSError:
                    pass
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(2)
            t = getattr(self, "_reader_thread", None)
            if t and t.is_alive():
                t.join(timeout=1)
        finally:
            if proc.stdout:
                try:
                    proc.stdout.close()
                except OSError:
                    pass


def _wrap(server: MCPServer, spec: dict) -> Tool:
    schema = spec.get("inputSchema") or {}
    required = set(schema.get("required", []))
    params = {}
    for k, v in (schema.get("properties") or {}).items():
        v = dict(v) if isinstance(v, dict) else {}
        t = v.get("type")
        nullable = False
        if isinstance(t, list):
            nullable = "null" in t
            non_null = [x for x in t if isinstance(x, str) and x != "null"]
            t = non_null[0] if len(non_null) == 1 else "string"
        elif not isinstance(t, str):
            variants = v.get("anyOf") or v.get("oneOf")
            if isinstance(variants, list):
                types = [x.get("type") for x in variants if isinstance(x, dict) and isinstance(x.get("type"), str)]
                nullable = "null" in types
                non_null = [x for x in types if x != "null"]
                t = non_null[0] if len(non_null) == 1 else "string"
                v.pop("anyOf", None)
                v.pop("oneOf", None)
            else:
                t = "string"
        v["type"] = t
        if nullable:
            v["nullable"] = True
        params[k] = {**v, "optional": k not in required}
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", f"{server.name}__{spec['name']}")[:64]
    desc = f"[MCP {server.name}] {spec.get('description', spec['name'])}"
    return Tool(name, desc, params, lambda **kw: server.call(spec["name"], kw), needs_approval=True)


def load_config(ws_root: Path) -> dict:
    for p in (ws_root / ".harness" / "mcp.json", Path.home() / ".config" / "harness" / "mcp.json"):
        if p.is_file():
            data = json.loads(p.read_text())
            if not isinstance(data, dict):
                raise MCPError("mcp.json phải là object")
            cfg = data.get("mcpServers") if "mcpServers" in data else data.get("servers")
            return {} if cfg is None else cfg
    return {}


def load_mcp_tools(ws_root: Path, warn=print) -> tuple[dict[str, Tool], list[MCPServer]]:
    tools: dict[str, Tool] = {}
    servers: list[MCPServer] = []
    try:
        cfg = load_config(ws_root)
    except (OSError, json.JSONDecodeError, MCPError) as e:
        warn(f"⚠ mcp.json lỗi: {e}")
        return tools, servers
    if not isinstance(cfg, dict):
        warn("⚠ mcp.json lỗi: mcpServers phải là object")
        return tools, servers
    for name, c in cfg.items():
        srv = None
        try:
            if not isinstance(c, dict):
                raise MCPError("cấu hình server phải là object")
            cmd = c["command"]
            if not isinstance(cmd, str) or not cmd.strip():
                raise MCPError("command phải là chuỗi khác rỗng")
            args = c.get("args")
            if args is not None and not isinstance(args, list):
                raise MCPError("args phải là mảng")
            env = c.get("env")
            if env is not None and not isinstance(env, dict):
                raise MCPError("env phải là object")
            srv = MCPServer(name, cmd, args, env, str(ws_root))
            srv.start()
            for spec in srv.list_tools():
                t = _wrap(srv, spec)
                tools[t.name] = t
            servers.append(srv)
        except (MCPError, KeyError, TypeError) as e:
            warn(f"⚠ MCP {name}: {e}")
            if srv:
                srv.stop()
    return tools, servers

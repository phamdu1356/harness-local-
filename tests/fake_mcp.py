"""MCP server giả (stdio, JSON-RPC theo dòng) để test client."""
import json
import sys

TOOLS = [{"name": "echo", "description": "Lặp lại văn bản",
          "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}, "times": {"type": "integer"}},
                          "required": ["text"]}},
         {"name": "boom", "description": "Luôn lỗi", "inputSchema": {"type": "object", "properties": {}}}]

for line in sys.stdin:
    msg = json.loads(line)
    if "id" not in msg:
        continue
    m, p = msg["method"], msg.get("params", {})
    if m == "initialize":
        res = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "fake"}}
    elif m == "tools/list":
        res = {"tools": TOOLS}
    elif m == "tools/call" and p["name"] == "echo":
        a = p["arguments"]
        res = {"content": [{"type": "text", "text": a["text"] * int(a.get("times", 1))}]}
    elif m == "tools/call":
        res = {"isError": True, "content": [{"type": "text", "text": "hỏng rồi"}]}
    else:
        print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "error": {"message": "không hỗ trợ"}}), flush=True)
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": res}), flush=True)

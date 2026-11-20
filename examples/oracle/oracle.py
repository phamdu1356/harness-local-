#!/usr/bin/env python3
"""Oracle nghiệm thu people-memory theo CONTRACT.md. Chạy app THẬT (trong sandbox Landlock), LLM giả và Telegram giả cục bộ.

    python3 oracle.py --list                # JSON các giai đoạn
    python3 oracle.py --upto 3              # chạy giai đoạn 1..3 (gồm cả hồi quy)
    python3 oracle.py --stage 4             # chỉ giai đoạn 4
    python3 oracle.py --upto 7 --only tên   # lọc theo tên check (debug)

Chạy từ thư mục dự án (hoặc --root). Mã thoát 0 nếu mọi giai đoạn được chọn đều đạt.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

for _parent in Path(__file__).resolve().parents:
    if (_parent / "harness" / "sandbox.py").is_file():
        sys.path.insert(0, str(_parent))
        break
try:
    from harness import sandbox  # noqa: E402
except ImportError:
    class sandbox:  # chạy độc lập khi copy oracle.py ra ngoài repo
        @staticmethod
        def available() -> bool: return False
        @staticmethod
        def wrap(cmd: str, root: str | Path) -> list[str]: return ["bash", "-c", cmd]

TOKEN = "123456:ABC-FAKE-TOKEN"
CHAT = "1001"
OTHER_CHAT = "2002"

STAGES = {
    1: ("Khởi động và giao diện", "Gói chạy được bằng `python3 -m people_memory`, đọc cấu hình từ biến môi trường, nghe trên 127.0.0.1, "
        "có /api/health, trang chủ HTML có ô nhập, dữ liệu nằm trong PM_DATA_DIR. Đọc CONTRACT.md mục 1 và 2."),
    2: ("Nhập một câu tự nhiên, lưu đúng người", "POST /api/message gọi LLM (OpenAI-compatible) lấy JSON trích xuất, lưu người và fact vào SQLite, "
        "bền qua khởi động lại, dưới 3 giây. Đọc CONTRACT.md mục 3 và 4."),
    3: ("Dữ liệu sạch và nhất quán", "Gộp biệt danh/tên (chuẩn hoá bỏ dấu), cập nhật fact đóng bản cũ thành superseded mà không xoá, "
        "/api/ask chỉ dùng fact active và không bịa khi không có người. Đọc CONTRACT.md mục 3 (quy tắc lưu) và 4."),
    4: ("Nhắc việc đúng lúc", "GET /api/reminders?now=...: sinh nhật (0..3 ngày, lặp hằng năm), sự kiện (0..1 ngày), lâu chưa liên lạc "
        "(ngưỡng theo quan hệ, số ngày chính xác), kế hoạch cuối tuần thứ Sáu 17h. Nội dung đúng số ngày. Đọc CONTRACT.md mục 5."),
    5: ("Soạn tin 3 sắc thái và gợi ý đi chơi", "/api/compose và /api/plan đưa hồ sơ (fact active, quan hệ, địa điểm đã đi) vào prompt, "
        "loại địa điểm đã đi, đúng 2 stop. Đọc CONTRACT.md mục 6."),
    6: ("Hàng đợi khi LLM tắt và sao lưu", "Tin vào hàng đợi bền khi LLM lỗi, xả bù khi LLM bật lại (tay và tự động), sao lưu SQLite hợp lệ, "
        "tự sao lưu khi khởi động. Đọc CONTRACT.md mục 8."),
    7: ("Telegram khoá một tài khoản và riêng tư", "Poll getUpdates với offset, chỉ xử lý chat id được phép, trả lời bằng sendMessage, đẩy nhắc việc một lần, "
        "không URL lạ, không lộ token. Đọc CONTRACT.md mục 9 và 10."),
}

# ---------------------------------------------------------------- tiện ích
def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower().replace("đ", "d"))
    return re.sub(r"\s+", " ", "".join(c for c in s if unicodedata.category(c) != "Mn")).strip()


class CheckFail(AssertionError):
    pass


HINTS: list[str] = []   # gợi ý chẩn đoán thu thập trong lúc chạy một check; đính kèm vào thông báo lỗi


def need(cond, what: str, sent=None, got=None):
    if not cond:
        if HINTS:
            got = f"{got if got is not None else ''}  >>> GỢI Ý CHẨN ĐOÁN: " + " ".join(dict.fromkeys(HINTS))
        raise CheckFail(json.dumps({"what": what, "sent": sent, "got": got}, ensure_ascii=False, default=str))


def clip(x, n=320) -> str:
    s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, default=str)
    return s if len(s) <= n else s[:n] + "…"


@dataclass
class Resp:
    status: int
    json: object
    text: str
    ctype: str
    elapsed: float


def http(method: str, url: str, body=None, timeout: float = 15) -> Resp:
    data = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status, raw, ctype = r.status, r.read(), r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        status, raw, ctype = e.code, e.read(), e.headers.get("Content-Type", "")
    except Exception as e:  # không kết nối được
        return Resp(0, None, f"{type(e).__name__}: {e}", "", time.time() - t0)
    text = raw.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    return Resp(status, parsed, text, ctype, time.time() - t0)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def db_contains(path: Path, needle: str) -> bool:
    """Schema-agnostic: có ô nào trong DB chứa chuỗi `needle` không."""
    con = sqlite3.connect(str(path))
    try:
        for (t,) in con.execute("select name from sqlite_master where type='table'"):
            for row in con.execute(f'select * from "{t}"'):
                if any(isinstance(c, str) and needle in c for c in row):
                    return True
    finally:
        con.close()
    return False


# ---------------------------------------------------------------- LLM giả
class FakeLLM:
    def __init__(self, port: int = 0):
        self.port = port or free_port()
        self.requests: list[str] = []
        self.extract: dict[str, tuple[object, str]] = {}   # câu -> (kết quả, kiểu: json|fence|garbage)
        self.plan_replies: list[dict] = []
        self.server = None

    def register(self, sentence: str, result, mode: str = "json"):
        self.extract[sentence] = (result, mode)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def start(self):
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def _send(self, obj, code=200):
                raw = json.dumps(obj, ensure_ascii=False).encode()
                self.send_response(code); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)

            def do_GET(self):
                self._send({"data": [{"id": "fake-model"}]})

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                txt = "\n".join(str(m.get("content", "")) for m in body.get("messages", []))
                fake.requests.append(txt)
                content = fake.reply(txt)
                self._send({"id": "x", "object": "chat.completion", "model": "fake-model",
                            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}]})

        class S(ThreadingHTTPServer):
            allow_reuse_address = True
            daemon_threads = True

        self.server = S(("127.0.0.1", self.port), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self):
        if self.server:
            self.server.shutdown(); self.server.server_close(); self.server = None

    def reply(self, txt: str) -> str:
        if '"gentle"' in txt and '"humor"' in txt:
            return json.dumps({"gentle": "G: mình nhắn nhẹ nhàng nhé", "firm": "F: mình nói thẳng nhưng lịch sự",
                               "humor": "H: đùa nhẹ cho đỡ căng"}, ensure_ascii=False)
        if '"stops"' in txt:
            if self.plan_replies:
                return json.dumps(self.plan_replies.pop(0), ensure_ascii=False)
            return json.dumps({"stops": [{"title": "Quán A", "reason": "hợp khẩu vị"}, {"title": "Cafe B", "reason": "yên tĩnh"}]},
                              ensure_ascii=False)
        for sentence, (result, mode) in self.extract.items():
            if sentence in txt:
                if mode == "garbage":
                    return "xin lỗi, tôi không thể trả về JSON"
                body = json.dumps(result, ensure_ascii=False)
                return f"```json\n{body}\n```" if mode == "fence" else body
        return "Câu trả lời giả dựa trên dữ liệu được cung cấp: " + txt[-200:]


# ---------------------------------------------------------------- Telegram giả
class FakeTelegram:
    def __init__(self):
        self.port = free_port()
        self.updates: list[dict] = []
        self.sent: list[dict] = []
        self.offsets: list[int] = []
        self.server = None
        self._id = 0

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def push(self, chat_id: str, text: str):
        self._id += 1
        self.updates.append({"update_id": self._id, "message": {"message_id": self._id, "chat": {"id": int(chat_id), "type": "private"},
                                                                  "from": {"id": int(chat_id)}, "text": text}})

    def sent_to(self, chat_id: str) -> list[dict]:
        return [m for m in self.sent if str(m.get("chat_id")) == str(chat_id)]

    def start(self):
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(n) or b"{}"
                try:
                    body = json.loads(raw)
                except json.JSONDecodeError:
                    body = {k: v[0] for k, v in __import__("urllib.parse").parse.parse_qs(raw.decode()).items()}
                if self.path.endswith("/getUpdates"):
                    off = int(body.get("offset", 0) or 0)
                    fake.offsets.append(off)
                    out = {"ok": True, "result": [u for u in fake.updates if u["update_id"] >= off]}
                elif self.path.endswith("/sendMessage"):
                    fake.sent.append(body)
                    out = {"ok": True, "result": {"message_id": len(fake.sent)}}
                else:
                    out = {"ok": True, "result": {}}
                data = json.dumps(out).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

            do_GET = do_POST

        class S(ThreadingHTTPServer):
            allow_reuse_address = True
            daemon_threads = True

        self.server = S(("127.0.0.1", self.port), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self):
        if self.server:
            self.server.shutdown(); self.server.server_close(); self.server = None


# ---------------------------------------------------------------- tiến trình app
class App:
    def __init__(self, root: Path, data_dir: Path, env: dict):
        self.root, self.data_dir, self.env = root, data_dir, env
        self.port = int(env["PM_PORT"])
        self.proc = None
        self.logf = root / ".oracle" / f"app-{self.port}.log"

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, wait: float = 12):
        self.logf.parent.mkdir(parents=True, exist_ok=True)
        full = {**os.environ, **self.env, "PYTHONDONTWRITEBYTECODE": "1"}
        cmd = sandbox.wrap("python3 -m people_memory", self.root) if sandbox.available() else ["bash", "-c", "python3 -m people_memory"]
        with open(self.logf, "ab") as logf:
            self.proc = subprocess.Popen(cmd, cwd=self.root, env=full, stdout=logf, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < wait:
            if self.proc.poll() is not None:
                need(False, "ứng dụng thoát ngay khi khởi động", "python3 -m people_memory", f"mã thoát {self.proc.returncode}; log: {self.tail()}")
            if http("GET", self.base + "/api/health", timeout=1).status == 200:
                return
            time.sleep(0.2)
        need(False, f"GET /api/health không trả lời trong {wait}s sau khi khởi động", "python3 -m people_memory", self.tail())

    def tail(self, n: int = 12) -> str:
        try:
            return " | ".join(self.logf.read_text(errors="replace").strip().splitlines()[-n:])[-700:]
        except OSError:
            return "(không có log)"

    def stop(self):
        if self.proc and self.proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.proc.pid, signal.SIGKILL)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    self.proc.wait(2)
        self.proc = None

    def restart(self):
        self.stop(); self.start()

    def get(self, path: str, **kw) -> Resp: return http("GET", self.base + path, **kw)
    def post(self, path: str, body=None, **kw) -> Resp: return http("POST", self.base + path, body if body is not None else {}, **kw)


class Scenario:
    def __init__(self, root: Path, n: int, llm=True, tg=False, chat=CHAT, interval="60", extra=None, token=TOKEN):
        self.dir = root / ".oracle" / f"s{n}"
        self.llm = FakeLLM() if llm else None
        self.tg = FakeTelegram() if tg else None
        port = free_port()
        env = {"PM_HOST": "127.0.0.1", "PM_PORT": str(port), "PM_DATA_DIR": str(self.dir / "data"), "PM_SCHEDULER_INTERVAL": interval,
               "PM_LLM_URL": (self.llm.url if self.llm else f"http://127.0.0.1:{free_port()}/v1"), "PM_LLM_TIMEOUT": "5",
               "PM_TELEGRAM_POLL": "0.2", "TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": ""}
        if tg:
            env.update({"TELEGRAM_BOT_TOKEN": token, "TELEGRAM_CHAT_ID": chat, "TELEGRAM_API_BASE": self.tg.base})
        env.update(extra or {})
        self.app = App(root, self.dir / "data", env)

    def __enter__(self):
        if self.llm: self.llm.start()
        if self.tg: self.tg.start()
        self.app.start()
        return self

    def __exit__(self, *exc):
        self.app.stop()
        if self.llm: self.llm.stop()
        if self.tg: self.tg.stop()

    def say(self, text: str, at: str | None = None) -> Resp:
        before = len(self.llm.requests) if self.llm else 0
        r = self.app.post("/api/message", {"text": text, **({"at": at} if at else {})})
        if self.llm and len(self.llm.requests) == before and self.llm.server is not None:
            HINTS.append("LLM giả KHÔNG nhận được yêu cầu nào: ứng dụng không hề gọi LLM. Phải dùng urllib gửi POST tới "
                         "{PM_LLM_URL}/chat/completions (đọc PM_LLM_URL từ biến môi trường) với text gốc của người dùng và yêu cầu LLM "
                         "trả JSON theo lược đồ CONTRACT.md mục 3; TUYỆT ĐỐI không tự trích xuất bằng regex/từ khoá.")
        elif self.llm and self.llm.server is not None and r.status not in (200,):
            HINTS.append("LLM có nhận yêu cầu nhưng ứng dụng không dùng được phản hồi: kiểm tra lại cách đọc choices[0].message.content, "
                         "bỏ khối ```json nếu có, rồi json.loads theo lược đồ trong CONTRACT.md.")
        return r

    def people(self) -> list[dict]:
        r = self.app.get("/api/people")
        return (r.json or {}).get("people", []) if isinstance(r.json, dict) else []

    def profile(self, pid: int) -> dict:
        r = self.app.get(f"/api/people/{pid}")
        need(r.status == 200 and isinstance(r.json, dict), f"GET /api/people/{pid} phải trả 200 và JSON", f"GET /api/people/{pid}", clip(r.text))
        return r.json

    def by_name(self, name: str) -> dict | None:
        return next((p for p in self.people() if norm(p.get("name", "")) == norm(name)), None)


# ---------------------------------------------------------------- dữ liệu mẫu
def F(kind, key, text, date_=None, recurring=None):
    return {"kind": kind, "key": key, "text": text, "date": date_, "recurring": recurring}


def P(name, facts=(), aliases=(), relation=None):
    return {"name": name, "aliases": list(aliases), "relation": relation, "facts": list(facts)}


S_A = "My bảo thích cuốn sách Y và dị ứng hải sản"
R_A = {"people": [P("My", [F("preference", "book:y", "thích cuốn sách Y"), F("allergy", "allergy:seafood", "dị ứng hải sản")], ["Mỹ"], "người yêu")]}
S_B = "Thằng Hùng mới chuyển trọ sang Cầu Giấy"
R_B = {"people": [P("Hùng", [F("info", "address", "mới chuyển trọ sang Cầu Giấy")], ["Thằng Hùng"], "bạn")]}
S_C = "Mẹ bị đau lưng, còn Nam thi môn Toán tuần sau"
R_C = {"people": [P("Mẹ", [F("health", "health:back", "bị đau lưng")], [], "mẹ"), P("Nam", [F("event", "exam:toan", "thi môn Toán", "2026-11-22")], [], "bạn")]}
S_D = "Mỹ vừa đổi việc sang công ty mới"
R_D = {"people": [P("Mỹ", [F("info", "job", "vừa đổi việc sang công ty mới")])]}
S_E = "em My thích hoa hồng"
R_E = {"people": [P("em My", [F("preference", "flower", "thích hoa hồng")], ["My"])]}
S_F = "my không thích ăn hành"
R_F = {"people": [P("my", [F("preference", "food:onion", "không thích ăn hành")])]}
S_G1 = "Lan thích ăn cay"
R_G1 = {"people": [P("Lan", [F("preference", "food:spicy", "thích ăn cay")], [], "bạn")]}
S_G2 = "Lan đang đau dạ dày phải kiêng cay"
R_G2 = {"people": [P("Lan", [F("preference", "food:spicy", "đang đau dạ dày, phải kiêng cay")])]}
S_BD = "Sinh nhật My là 20/12"
R_BD = {"people": [P("My", [F("event", "birthday", "sinh nhật", "2026-12-20", "yearly")], [], "người yêu")]}
S_EX = "My thi môn Toán ngày mai"
R_EX = {"people": [P("My", [F("event", "exam:toan", "thi môn Toán", "2026-11-16")], [], "người yêu")]}
S_PL = "Hôm qua đi ăn với My ở Quán Phở Thìn"
R_PL = {"people": [P("My", [F("place", "place:quan-pho-thin", "Quán Phở Thìn")], [], "người yêu")]}
S_ME = "Gọi điện cho mẹ"
R_ME = {"people": [P("Mẹ", [], [], "mẹ")]}
S_NAM = "Đi cafe với Nam"
R_NAM = {"people": [P("Nam", [], [], "bạn")]}
S_HOA = "Nhắn tin cho Hoa"
R_HOA = {"people": [P("Hoa", [], [], "bạn")]}
S_YEU = "Đi ăn tối với My"
R_YEU = {"people": [P("My", [], [], "người yêu")]}
S_H = "Hùng vừa mua xe mới màu đỏ"
R_H = {"people": [P("Hùng", [F("info", "car", "mới mua xe màu đỏ")])]}

REG = [(S_A, R_A), (S_B, R_B), (S_C, R_C), (S_D, R_D), (S_E, R_E), (S_F, R_F), (S_G1, R_G1), (S_G2, R_G2), (S_BD, R_BD), (S_EX, R_EX),
       (S_PL, R_PL), (S_ME, R_ME), (S_NAM, R_NAM), (S_HOA, R_HOA), (S_YEU, R_YEU)]


def load(sc: Scenario, *pairs):
    for s, r in pairs:
        sc.llm.register(s, r)


# ---------------------------------------------------------------- các check
CHECKS: list[tuple[int, str, callable]] = []


def check(stage: int, name: str):
    def deco(fn):
        CHECKS.append((stage, name, fn))
        return fn
    return deco


# ===== Giai đoạn 1
@check(1, "chạy được bằng `python3 -m people_memory` và /api/health trả {\"ok\": true}")
def c1_boot(root, n):
    with Scenario(root, n) as sc:
        r = sc.app.get("/api/health")
        need(r.status == 200 and isinstance(r.json, dict) and r.json.get("ok") is True, "GET /api/health phải trả 200 {\"ok\": true}", "GET /api/health", clip(r.text))


@check(1, "tạo PM_DATA_DIR và file people.db; GET /api/people ban đầu rỗng")
def c1_data(root, n):
    with Scenario(root, n) as sc:
        r = sc.app.get("/api/people")
        need(r.status == 200 and isinstance(r.json, dict) and r.json.get("people") == [], "GET /api/people khi chưa có dữ liệu phải là {\"people\": []}", "GET /api/people", clip(r.text))
        need((sc.app.data_dir / "people.db").exists(), "phải tạo file PM_DATA_DIR/people.db (SQLite)", f"ls {sc.app.data_dir}", sorted(p.name for p in sc.app.data_dir.glob("*")) if sc.app.data_dir.exists() else "thư mục không tồn tại")


@check(1, "trang chủ GET / là HTML có ô nhập và form/nút gửi")
def c1_home(root, n):
    with Scenario(root, n) as sc:
        r = sc.app.get("/")
        low = r.text.lower()
        need(r.status == 200 and "text/html" in r.ctype.lower(), "GET / phải trả 200 với Content-Type text/html", "GET /", f"{r.status} {r.ctype} {clip(r.text, 120)}")
        need(("<textarea" in low or "<input" in low) and ("<form" in low or "<button" in low), "trang chủ phải có <textarea>/<input> và <form>/<button>", "GET /", clip(r.text, 200))


@check(1, "chỉ nghe trên 127.0.0.1 theo mặc định (không mở ra mạng LAN)")
def c1_bind(root, n):
    ip = None
    with contextlib.suppress(OSError):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1)); ip = s.getsockname()[0]
    if not ip or ip.startswith("127."):
        return  # máy không có địa chỉ ngoài: bỏ qua
    port = free_port()
    sc = Scenario(root, n, extra={"PM_HOST": "", "PM_PORT": str(port)})
    sc.app.env.pop("PM_HOST", None)
    sc.app.port = port
    if sc.llm: sc.llm.start()
    try:
        sc.app.start()
        with socket.socket() as s:
            s.settimeout(1.5)
            opened = s.connect_ex((ip, port)) == 0
        need(not opened, f"khi không đặt PM_HOST, ứng dụng không được nghe trên {ip}:{port} (mặc định phải là 127.0.0.1)", f"kết nối tới {ip}:{port}", "kết nối được")
    finally:
        sc.app.stop()
        if sc.llm: sc.llm.stop()


# ===== Giai đoạn 2
@check(2, "một câu tự nhiên -> 200 saved, tạo đúng người, lưu đủ fact, dưới 3 giây")
def c2_save(root, n):
    with Scenario(root, n) as sc:
        load(sc, (S_A, R_A))
        r = sc.say(S_A)
        need(r.status == 200 and isinstance(r.json, dict) and r.json.get("status") == "saved", "POST /api/message phải trả 200 {\"status\": \"saved\", ...}", {"text": S_A}, f"{r.status} {clip(r.text)}")
        need(r.elapsed < 3, f"phải xong dưới 3 giây, mất {r.elapsed:.2f}s", {"text": S_A}, f"{r.elapsed:.2f}s")
        ppl = r.json.get("people") or []
        need(len(ppl) == 1 and norm(ppl[0].get("name", "")) == "my" and ppl[0].get("created") is True, "phản hồi phải có people=[{id, name: \"My\", created: true}]", {"text": S_A}, clip(r.json))
        need(r.json.get("facts_added") == 2, "facts_added phải bằng 2", {"text": S_A}, clip(r.json))
        pr = sc.profile(ppl[0]["id"])
        texts = {f["text"]: f for f in pr.get("facts", [])}
        need("dị ứng hải sản" in texts and "thích cuốn sách Y" in texts, "hồ sơ phải có fact 'dị ứng hải sản' và 'thích cuốn sách Y'", f"GET /api/people/{ppl[0]['id']}", clip(pr))
        need(all(f.get("status") == "active" for f in texts.values()), "fact mới phải có status \"active\"", None, clip(pr.get("facts")))
        need(pr.get("relation") == "người yêu", "relation phải là \"người yêu\" theo kết quả LLM", None, clip(pr))
        need(any(f.get("kind") == "allergy" and f.get("key") == "allergy:seafood" for f in pr["facts"]), "fact phải giữ kind và key từ LLM", None, clip(pr["facts"]))


@check(2, "text rỗng -> HTTP 400")
def c2_empty(root, n):
    with Scenario(root, n) as sc:
        for body in ({"text": ""}, {"text": "   "}, {}):
            r = sc.app.post("/api/message", body)
            need(r.status == 400, "text rỗng/thiếu phải trả HTTP 400", body, f"{r.status} {clip(r.text, 100)}")


@check(2, "dữ liệu còn nguyên sau khi khởi động lại")
def c2_persist(root, n):
    with Scenario(root, n) as sc:
        load(sc, (S_A, R_A))
        need(sc.say(S_A).status == 200, "lưu câu đầu phải thành công", {"text": S_A})
        sc.app.restart()
        me = sc.by_name("My")
        need(me is not None, "sau khi khởi động lại GET /api/people vẫn phải có My", "GET /api/people (sau restart)", clip(sc.people()))
        need(any(f["text"] == "dị ứng hải sản" for f in sc.profile(me["id"])["facts"]), "fact vẫn phải còn sau khi khởi động lại", None, clip(sc.profile(me["id"])))


@check(2, "chịu được JSON bọc trong ```json ... ```")
def c2_fence(root, n):
    with Scenario(root, n) as sc:
        sc.llm.register(S_B, R_B, mode="fence")
        r = sc.say(S_B)
        need(r.status == 200 and r.json.get("status") == "saved", "LLM trả JSON trong khối ```json phải vẫn lưu được", {"text": S_B}, f"{r.status} {clip(r.text)}")
        need(sc.by_name("Hùng") is not None, "phải tạo người Hùng", None, clip(sc.people()))


@check(2, "`at` được lưu làm last_interaction_at")
def c2_at(root, n):
    with Scenario(root, n) as sc:
        load(sc, (S_A, R_A))
        sc.say(S_A, at="2026-11-01T10:00:00")
        me = sc.by_name("My")
        need(me is not None and str(me.get("last_interaction_at", "")).startswith("2026-11-01T10:00"), "last_interaction_at phải bằng `at` (2026-11-01T10:00:00)", {"text": S_A, "at": "2026-11-01T10:00:00"}, clip(me))


@check(2, "một tin nhắc nhiều người -> tạo nhiều người")
def c2_multi(root, n):
    with Scenario(root, n) as sc:
        load(sc, (S_C, R_C))
        r = sc.say(S_C)
        need(r.status == 200 and len(r.json.get("people", [])) == 2, "phải trả 2 người trong `people`", {"text": S_C}, clip(r.text))
        names = {norm(p["name"]) for p in sc.people()}
        need(names == {"me", "nam"}, "GET /api/people phải có Mẹ và Nam", None, clip(sc.people()))


# ===== Giai đoạn 3
@check(3, "biệt danh/tên trùng sau chuẩn hoá được gộp, không tạo người mới")
def c3_alias(root, n):
    with Scenario(root, n) as sc:
        load(sc, (S_A, R_A), (S_D, R_D), (S_E, R_E), (S_F, R_F))
        sc.say(S_A)
        sc.say(S_D)   # tên "Mỹ" khớp biệt danh "Mỹ" đã lưu
        need(len(sc.people()) == 1, "tên 'Mỹ' trùng biệt danh của My phải được gộp (vẫn 1 người)", {"text": S_D}, clip(sc.people()))
        sc.say(S_E)   # tên "em My", biệt danh "My" trùng tên
        need(len(sc.people()) == 1, "tên 'em My' có biệt danh 'My' trùng tên My phải được gộp (vẫn 1 người)", {"text": S_E}, clip(sc.people()))
        sc.say(S_F)   # tên "my" viết thường
        ppl = sc.people()
        need(len(ppl) == 1, "tên 'my' (khác hoa/thường) phải được gộp (vẫn 1 người)", {"text": S_F}, clip(ppl))
        al = {norm(a) for a in ppl[0].get("aliases", [])}
        need("em my" in al, "biệt danh 'em My' phải được thêm vào aliases", None, clip(ppl[0]))
        texts = {f["text"] for f in sc.profile(ppl[0]["id"])["facts"]}
        need({"vừa đổi việc sang công ty mới", "thích hoa hồng", "không thích ăn hành"} <= texts, "fact từ các tin gộp phải nằm trong hồ sơ của một người", None, sorted(texts))


@check(3, "cập nhật fact: bản cũ thành superseded và được giữ lại; lặp lại không nhân đôi")
def c3_supersede(root, n):
    with Scenario(root, n) as sc:
        load(sc, (S_G1, R_G1), (S_G2, R_G2))
        sc.say(S_G1)
        sc.say(S_G2)
        lan = sc.by_name("Lan")
        need(lan is not None, "phải có người Lan", None, clip(sc.people()))
        facts = [f for f in sc.profile(lan["id"])["facts"] if f.get("key") == "food:spicy"]
        need(len(facts) == 2, "phải có đúng 2 fact cùng key food:spicy (1 cũ superseded + 1 mới active)", None, clip(facts))
        old = next((f for f in facts if f["text"] == "thích ăn cay"), None)
        new = next((f for f in facts if f["text"] == "đang đau dạ dày, phải kiêng cay"), None)
        need(old and old.get("status") == "superseded" and old.get("superseded_at"), "fact cũ 'thích ăn cay' phải là superseded và có superseded_at (không được xoá)", None, clip(facts))
        need(new and new.get("status") == "active", "fact mới phải là active", None, clip(facts))
        sc.say(S_G2)
        facts2 = [f for f in sc.profile(lan["id"])["facts"] if f.get("key") == "food:spicy"]
        need(len(facts2) == 2, "gửi lại cùng nội dung không được tạo thêm fact", {"text": S_G2}, clip(facts2))


@check(3, "/api/ask với người đã biết: trả fact ACTIVE đúng người")
def c3_ask(root, n):
    with Scenario(root, n) as sc:
        load(sc, (S_A, R_A), (S_G1, R_G1), (S_G2, R_G2))
        sc.say(S_A); sc.say(S_G1); sc.say(S_G2)
        r = sc.app.post("/api/ask", {"question": "My dị ứng gì?"})
        need(r.status == 200 and isinstance(r.json, dict), "POST /api/ask phải trả 200 JSON", {"question": "My dị ứng gì?"}, f"{r.status} {clip(r.text)}")
        texts = [f.get("text") for f in r.json.get("facts", [])]
        need("dị ứng hải sản" in texts, "facts phải chứa fact active 'dị ứng hải sản' của My", None, clip(r.json))
        need(isinstance(r.json.get("answer"), str) and r.json["answer"].strip(), "answer phải là chuỗi không rỗng", None, clip(r.json))
        r2 = sc.app.post("/api/ask", {"question": "Lan thích ăn gì?"})
        need(r2.status == 200 and isinstance(r2.json, dict), "POST /api/ask (Lan) phải trả 200 JSON", {"question": "Lan thích ăn gì?"}, f"{r2.status} {clip(r2.text)}")
        t2 = [f.get("text") for f in r2.json.get("facts", [])]
        need("đang đau dạ dày, phải kiêng cay" in t2 and "thích ăn cay" not in t2, "ask phải chỉ trả fact ACTIVE (không trả bản cũ superseded)", None, clip(r2.json))
        need(all(f.get("status") == "active" for f in r.json.get("facts", []) + r2.json.get("facts", [])), "mọi fact trả về phải active", None, clip(r2.json.get("facts")))


@check(3, "/api/ask với người không có: không bịa và không gọi LLM")
def c3_ask_unknown(root, n):
    with Scenario(root, n) as sc:
        load(sc, (S_A, R_A))
        sc.say(S_A)
        before = len(sc.llm.requests)
        r = sc.app.post("/api/ask", {"question": "Bác Ba Lô thích gì?"})
        need(r.status == 200 and r.json.get("people") == [] and r.json.get("facts") == [], "người không có thì people=[] và facts=[]", {"question": "Bác Ba Lô thích gì?"}, clip(r.text))
        need("chưa có thông tin" in norm(r.json.get("answer", "")).replace("chua co thong tin", "chưa có thông tin") or "chua co thong tin" in norm(r.json.get("answer", "")),
             "answer phải nói rõ là chưa có thông tin (không bịa)", None, clip(r.json))
        need(len(sc.llm.requests) == before, "không được gọi LLM khi không có người nào được nhắc tới", None, f"LLM nhận thêm {len(sc.llm.requests) - before} yêu cầu")


# ===== Giai đoạn 4
def _rem(sc, now):
    r = sc.app.get(f"/api/reminders?now={now}")
    need(r.status == 200 and isinstance(r.json, dict) and isinstance(r.json.get("reminders"), list), "GET /api/reminders?now=... phải trả {\"reminders\": [...]}", f"GET /api/reminders?now={now}", f"{r.status} {clip(r.text)}")
    return r.json["reminders"]


@check(4, "sinh nhật: nhắc khi còn 0..3 ngày, đúng số ngày, lặp hằng năm")
def c4_birthday(root, n):
    with Scenario(root, n) as sc:
        load(sc, (S_BD, R_BD))
        sc.say(S_BD, at="2026-11-10T09:00:00")
        me = sc.by_name("My")["id"]
        b = [x for x in _rem(sc, "2026-12-17T09:00:00") if x.get("type") == "birthday"]
        need(len(b) == 1 and b[0].get("days") == 3 and b[0].get("due") == "2026-12-20" and b[0].get("person_id") == me, "còn 3 ngày: phải có đúng 1 nhắc birthday, days=3, due=2026-12-20, person_id của My", "now=2026-12-17T09:00:00", clip(b))
        m = b[0].get("message", "")
        need("My" in m and "3 ngày" in m and "tối nay" not in m.lower(), "message phải chứa tên 'My' và '3 ngày', không được nói 'tối nay'", "now=2026-12-17T09:00:00", m)
        b0 = [x for x in _rem(sc, "2026-12-20T08:00:00") if x.get("type") == "birthday"]
        need(len(b0) == 1 and b0[0].get("days") == 0 and "hôm nay" in b0[0].get("message", "").lower(), "đúng ngày sinh nhật: days=0 và message chứa 'hôm nay'", "now=2026-12-20T08:00:00", clip(b0))
        need(not [x for x in _rem(sc, "2026-12-10T09:00:00") if x.get("type") == "birthday"], "còn 10 ngày thì chưa nhắc", "now=2026-12-10T09:00:00", clip(_rem(sc, "2026-12-10T09:00:00")))
        nx = [x for x in _rem(sc, "2027-12-18T09:00:00") if x.get("type") == "birthday"]
        need(len(nx) == 1 and nx[0].get("due") == "2027-12-20" and nx[0].get("days") == 2, "sinh nhật lặp hằng năm: năm sau còn 2 ngày, due=2027-12-20", "now=2027-12-18T09:00:00", clip(nx))
        a, b2 = _rem(sc, "2026-12-17T09:00:00"), _rem(sc, "2026-12-17T09:00:00")
        need(a == b2, "gọi hai lần cùng `now` phải cho cùng kết quả (idempotent)", "now=2026-12-17T09:00:00 ×2", f"{clip(a)} != {clip(b2)}")


@check(4, "sự kiện (kỳ thi) ngày mai: nhắc hỏi thăm, đúng ngày")
def c4_event(root, n):
    with Scenario(root, n) as sc:
        load(sc, (S_EX, R_EX))
        sc.say(S_EX, at="2026-11-10T09:00:00")
        e = [x for x in _rem(sc, "2026-11-15T09:00:00") if x.get("type") == "event"]
        need(len(e) == 1 and e[0].get("days") == 1 and "My" in e[0].get("message", "") and "thi môn Toán" in e[0].get("message", ""), "còn 1 ngày: 1 nhắc event, days=1, message chứa 'My' và 'thi môn Toán'", "now=2026-11-15T09:00:00", clip(e))
        need(not [x for x in _rem(sc, "2026-11-12T09:00:00") if x.get("type") == "event"], "còn 4 ngày thì chưa nhắc sự kiện", "now=2026-11-12T09:00:00", clip(_rem(sc, "2026-11-12T09:00:00")))
        need(not [x for x in _rem(sc, "2026-11-17T09:00:00") if x.get("type") == "event"], "sự kiện đã qua thì không nhắc", "now=2026-11-17T09:00:00", clip(_rem(sc, "2026-11-17T09:00:00")))


@check(4, "lâu chưa liên lạc: ngưỡng theo quan hệ, số ngày chính xác, không làm phiền")
def c4_nocontact(root, n):
    with Scenario(root, n) as sc:
        load(sc, (S_ME, R_ME), (S_NAM, R_NAM), (S_HOA, R_HOA), (S_YEU, R_YEU))
        sc.say(S_ME, at="2026-10-01T09:00:00")    # mẹ: 45 ngày (>21)
        sc.say(S_NAM, at="2026-09-01T09:00:00")   # bạn: 75 ngày (>60)
        sc.say(S_HOA, at="2026-10-20T09:00:00")   # bạn: 26 ngày (<60)
        sc.say(S_YEU, at="2026-11-05T09:00:00")   # người yêu: 10 ngày (>7)
        rem = [x for x in _rem(sc, "2026-11-15T09:00:00") if x.get("type") == "no_contact"]
        by = {sc.profile(x["person_id"])["name"]: x for x in rem}
        need("Mẹ" in by and by["Mẹ"].get("days") == 45 and "45" in by["Mẹ"].get("message", ""), "Mẹ (ngưỡng 21) không liên lạc 45 ngày: phải nhắc, days=45, message chứa '45'", "now=2026-11-15T09:00:00", clip(rem))
        need("Nam" in by and by["Nam"].get("days") == 75, "Nam (bạn, ngưỡng 60) 75 ngày: phải nhắc, days=75", None, clip(rem))
        need("Hoa" not in by, "Hoa (bạn) mới 26 ngày chưa quá ngưỡng 60: không được nhắc", None, clip(rem))
        need("My" in by and by["My"].get("days") == 10, "My (người yêu, ngưỡng 7) 10 ngày: phải nhắc, days=10", None, clip(rem))
        need(len(rem) == 3, f"chỉ có 3 nhắc no_contact, nhận {len(rem)}", None, clip(rem))


@check(4, "kế hoạch cuối tuần: đúng 1 nhắc vào tối thứ Sáu từ 17h, không ngày khác")
def c4_weekly(root, n):
    d = date(2026, 11, 9)
    while d.weekday() != 4:
        d += timedelta(days=1)
    with Scenario(root, n) as sc:
        w = [x for x in _rem(sc, f"{d}T18:00:00") if x.get("type") == "weekly_plan"]
        need(len(w) == 1 and w[0].get("person_id") is None and w[0].get("message"), f"thứ Sáu {d} 18:00: phải có đúng 1 nhắc weekly_plan, person_id null, message không rỗng", f"now={d}T18:00:00", clip(w))
        need(not [x for x in _rem(sc, f"{d}T10:00:00") if x.get("type") == "weekly_plan"], "thứ Sáu 10:00 (trước 17h) chưa nhắc", f"now={d}T10:00:00", "có nhắc")
        nd = d + timedelta(days=1)
        need(not [x for x in _rem(sc, f"{nd}T18:00:00") if x.get("type") == "weekly_plan"], "thứ Bảy không nhắc weekly_plan", f"now={nd}T18:00:00", "có nhắc")


# ===== Giai đoạn 5
@check(5, "compose: 3 sắc thái, prompt chứa hồ sơ và tình huống")
def c5_compose(root, n):
    with Scenario(root, n) as sc:
        load(sc, (S_A, R_A))
        sc.say(S_A)
        sit = "Đang bực lắm, muốn nhắn là tại sao lúc nào cũng trễ hẹn"
        r = sc.app.post("/api/compose", {"person": "My", "situation": sit})
        need(r.status == 200 and isinstance(r.json, dict) and set((r.json.get("options") or {}).keys()) == {"gentle", "firm", "humor"}, "POST /api/compose phải trả 200 {\"options\": {gentle, firm, humor}}", {"person": "My", "situation": sit}, f"{r.status} {clip(r.text)}")
        need(all(isinstance(v, str) and v.strip() for v in r.json["options"].values()), "cả 3 lựa chọn phải là chuỗi không rỗng", None, clip(r.json))
        last = sc.llm.requests[-1]
        need("dị ứng hải sản" in last and "người yêu" in last and "trễ hẹn" in last, "prompt gửi LLM phải chứa fact active ('dị ứng hải sản'), quan hệ ('người yêu') và tình huống", None, "thiếu: " + ", ".join(k for k in ("dị ứng hải sản", "người yêu", "trễ hẹn") if k not in last))
        need('"gentle"' in last and '"firm"' in last and '"humor"' in last, "prompt phải yêu cầu LLM trả JSON có khoá gentle, firm, humor", None, "prompt thiếu tên khoá")
        r404 = sc.app.post("/api/compose", {"person": "Không Ai Cả", "situation": "x"})
        need(r404.status == 404, "người không tồn tại phải trả 404", {"person": "Không Ai Cả"}, f"{r404.status}")


@check(5, "plan: 2 chặng, prompt có sở thích/dị ứng/yêu cầu/thời tiết, loại địa điểm đã đi")
def c5_plan(root, n):
    with Scenario(root, n) as sc:
        load(sc, (S_A, R_A), (S_F, R_F), (S_PL, R_PL))
        sc.say(S_A); sc.say(S_F); sc.say(S_PL)
        sc.llm.plan_replies = [
            {"stops": [{"title": "Quán Phở Thìn", "reason": "ngon"}, {"title": "Cafe Muối", "reason": "yên tĩnh"}]},
            {"stops": [{"title": "Bún chả Hương Liên", "reason": "hợp khẩu vị"}, {"title": "Cafe Muối", "reason": "yên tĩnh"}]},
        ]
        req = "Tối thứ 7 đi đâu quanh Hà Đông, trời lạnh, dưới 500k"
        r = sc.app.post("/api/plan", {"person": "My", "request": req, "weather": "lạnh"})
        need(r.status == 200 and isinstance(r.json, dict), "POST /api/plan phải trả 200 JSON", {"person": "My", "request": req}, f"{r.status} {clip(r.text)}")
        stops = r.json.get("stops", [])
        need(len(stops) == 2 and all(s.get("title") and s.get("reason") for s in stops), "phải có đúng 2 stop, mỗi stop có title và reason", None, clip(r.json))
        need(all(norm(s["title"]) != norm("Quán Phở Thìn") for s in stops), "stop trùng địa điểm đã đi (Quán Phở Thìn) phải bị loại; kết quả không được chứa nó", None, clip(stops))
        need("Quán Phở Thìn" in r.json.get("excluded", []), "excluded phải liệt kê địa điểm đã đi 'Quán Phở Thìn'", None, clip(r.json))
        first = sc.llm.requests[[i for i, t in enumerate(sc.llm.requests) if '"stops"' in t][0]]
        miss = [k for k in ("không thích ăn hành", "dị ứng hải sản", "Quán Phở Thìn", "Hà Đông", "lạnh") if k not in first]
        need(not miss, "prompt gửi LLM phải chứa sở thích/dị ứng active, địa điểm đã đi, yêu cầu và thời tiết", None, "thiếu: " + ", ".join(miss))
        need(len([t for t in sc.llm.requests if '"stops"' in t]) >= 2, "khi LLM gợi ý trùng địa điểm đã đi, phải hỏi LLM lại", None, "chỉ gọi 1 lần")
        r404 = sc.app.post("/api/plan", {"person": "Không Ai Cả", "request": "x"})
        need(r404.status == 404, "người không tồn tại phải trả 404", {"person": "Không Ai Cả"}, f"{r404.status}")


@check(5, "compose/plan khi LLM tắt -> 503")
def c5_down(root, n):
    with Scenario(root, n) as sc:
        load(sc, (S_A, R_A))
        sc.say(S_A)
        sc.llm.stop()
        r1 = sc.app.post("/api/compose", {"person": "My", "situation": "x"}, timeout=20)
        need(r1.status == 503, "compose khi LLM tắt phải trả 503", {"person": "My", "situation": "x"}, f"{r1.status} {clip(r1.text, 100)}")
        r2 = sc.app.post("/api/plan", {"person": "My", "request": "x"}, timeout=20)
        need(r2.status == 503, "plan khi LLM tắt phải trả 503", {"person": "My", "request": "x"}, f"{r2.status} {clip(r2.text, 100)}")
        sc.llm.start()


# ===== Giai đoạn 6
@check(6, "LLM tắt: tin vào hàng đợi (202), không mất, còn sau khởi động lại, xả bù khi LLM bật lại")
def c6_queue(root, n):
    with Scenario(root, n) as sc:
        load(sc, (S_A, R_A))
        sc.llm.stop()
        r = sc.say(S_A, at="2026-11-01T10:00:00")
        need(r.status == 202 and r.json.get("status") == "queued" and "queue_id" in r.json, "LLM tắt phải trả 202 {\"status\": \"queued\", \"queue_id\": ...}", {"text": S_A}, f"{r.status} {clip(r.text)}")
        q = sc.app.get("/api/queue")
        need(q.status == 200 and q.json.get("pending") == 1 and S_A in json.dumps(q.json, ensure_ascii=False), "GET /api/queue phải cho pending=1 và liệt kê tin", None, clip(q.text))
        sc.app.restart()
        q = sc.app.get("/api/queue")
        need(q.json.get("pending") == 1, "hàng đợi phải còn nguyên sau khi khởi động lại", None, clip(q.text))
        d0 = sc.app.post("/api/queue/drain", timeout=30)
        need(d0.status == 200 and d0.json.get("processed") == 0 and d0.json.get("remaining") == 1, "xả khi LLM còn tắt: processed=0, remaining=1 (giữ lại tin)", None, clip(d0.text))
        att = sc.app.get("/api/queue").json["items"][0].get("attempts", 0)
        need(att >= 1, "attempts phải tăng sau lần xả thất bại", None, clip(sc.app.get("/api/queue").text))
        sc.llm.start()
        d = sc.app.post("/api/queue/drain", timeout=30)
        need(d.status == 200 and d.json.get("processed") == 1 and d.json.get("remaining") == 0, "LLM bật lại: xả phải processed=1, remaining=0", None, clip(d.text))
        me = sc.by_name("My")
        need(me is not None and str(me.get("last_interaction_at", "")).startswith("2026-11-01T10:00"), "tin được xử lý bù phải lưu đúng người và dùng `at` đã lưu", None, clip(sc.people()))
        need(sc.app.get("/api/queue").json.get("pending") == 0, "hàng đợi phải trống sau khi xả", None, clip(sc.app.get("/api/queue").text))


@check(6, "JSON hỏng từ LLM -> tin vào hàng đợi, không mất")
def c6_garbage(root, n):
    with Scenario(root, n) as sc:
        sc.llm.register(S_H, R_H, mode="garbage")
        r = sc.say(S_H)
        need(r.status == 202 and r.json.get("status") == "queued", "LLM trả không phải JSON phải coi là lỗi và xếp hàng (202 queued)", {"text": S_H}, f"{r.status} {clip(r.text)}")
        need(sc.app.get("/api/queue").json.get("pending") == 1, "tin phải nằm trong hàng đợi", None, clip(sc.app.get("/api/queue").text))


@check(6, "luồng nền tự xả hàng đợi khi LLM bật lại")
def c6_auto(root, n):
    with Scenario(root, n, interval="1") as sc:
        load(sc, (S_A, R_A))
        sc.llm.stop()
        need(sc.say(S_A).status == 202, "LLM tắt phải trả 202", {"text": S_A})
        sc.llm.start()
        t0 = time.time()
        while time.time() - t0 < 8:
            if sc.app.get("/api/queue").json.get("pending") == 0 and sc.by_name("My"):
                return
            time.sleep(0.4)
        need(False, "trong 8 giây (PM_SCHEDULER_INTERVAL=1) luồng nền phải tự xả hàng đợi và lưu My", None, clip(sc.app.get("/api/queue").text))


@check(6, "sao lưu thủ công: file SQLite hợp lệ chứa đủ dữ liệu, có trong /api/backups")
def c6_backup(root, n):
    with Scenario(root, n) as sc:
        load(sc, (S_A, R_A))
        sc.say(S_A)
        r = sc.app.post("/api/backup")
        need(r.status == 200 and isinstance(r.json, dict) and r.json.get("path"), "POST /api/backup phải trả 200 {\"path\": ...}", None, f"{r.status} {clip(r.text)}")
        p = Path(r.json["path"])
        need(p.exists() and p.parent.name == "backups" and re.fullmatch(r"people-\d{8}-\d{6}\.db", p.name), "file phải nằm trong PM_DATA_DIR/backups/ và tên dạng people-YYYYmmdd-HHMMSS.db", None, str(p))
        con = sqlite3.connect(str(p)); ok = con.execute("pragma integrity_check").fetchone()[0]; con.close()
        need(ok == "ok", "file backup phải là SQLite hợp lệ", None, ok)
        need(db_contains(p, "dị ứng hải sản"), "bản sao lưu phải chứa dữ liệu hiện tại ('dị ứng hải sản')", None, "không tìm thấy")
        lst = sc.app.get("/api/backups")
        need(lst.status == 200 and any(b.get("path") == str(p) and b.get("size", 0) > 0 for b in lst.json.get("backups", [])), "GET /api/backups phải liệt kê file vừa tạo kèm size", None, clip(lst.text))


@check(6, "tự sao lưu khi khởi động; khôi phục bằng cách chép đè people.db")
def c6_auto_backup_restore(root, n):
    with Scenario(root, n) as sc:
        need(any(True for _ in sc.app.get("/api/backups").json.get("backups", [])), "khi khởi động mà chưa có bản sao lưu nào thì phải tự sao lưu ngay", "GET /api/backups ngay sau khi khởi động", clip(sc.app.get("/api/backups").text))
        load(sc, (S_A, R_A))
        sc.say(S_A)
        path = Path(sc.app.post("/api/backup").json["path"])
        sc.app.stop()
        shutil.copy(path, sc.app.data_dir / "people.db")
        sc.app.start()
        need(sc.by_name("My") is not None, "chép bản sao lưu đè lên people.db rồi khởi động lại phải khôi phục được dữ liệu", None, clip(sc.people()))


# ===== Giai đoạn 7
def _wait(cond, t=6.0):
    t0 = time.time()
    while time.time() - t0 < t:
        if cond():
            return True
        time.sleep(0.2)
    return False


@check(7, "Telegram: nhận tin từ chat được phép, lưu đúng người và trả lời bằng sendMessage")
def c7_flow(root, n):
    with Scenario(root, n, tg=True) as sc:
        load(sc, (S_A, R_A))
        sc.tg.push(CHAT, S_A)
        need(_wait(lambda: sc.by_name("My")), "trong 6 giây tin Telegram từ chat được phép phải được lưu (My)", f"update chat {CHAT}: {S_A}", clip(sc.people()))
        need(_wait(lambda: sc.tg.sent_to(CHAT)), "phải trả lời bằng sendMessage tới chat được phép", None, clip(sc.tg.sent))
        time.sleep(1.0)
        need(len(sc.tg.sent_to(CHAT)) == 1, "mỗi update chỉ trả lời đúng 1 lần (phải dùng offset, không xử lý lại)", None, f"{len(sc.tg.sent_to(CHAT))} tin trả lời; offsets đã gửi: {sc.tg.offsets[-5:]}")
        need(any(o > 1 for o in sc.tg.offsets), "phải gửi offset = update_id lớn nhất + 1 trong getUpdates", None, f"offsets={sc.tg.offsets[-8:]}")
        need(len(sc.people()) == 1, "chỉ có 1 người", None, clip(sc.people()))


@check(7, "Telegram: bỏ qua hoàn toàn chat id khác (không lưu, không gọi LLM, không trả lời)")
def c7_lock(root, n):
    with Scenario(root, n, tg=True) as sc:
        load(sc, (S_A, R_A), (S_B, R_B))
        sc.tg.push(OTHER_CHAT, S_B)
        time.sleep(2.0)
        need(not sc.people(), "tin từ chat id lạ không được lưu", f"update chat {OTHER_CHAT}: {S_B}", clip(sc.people()))
        need(not sc.tg.sent_to(OTHER_CHAT) and not sc.tg.sent, "không được trả lời chat id lạ", None, clip(sc.tg.sent))
        need(not any(S_B in t for t in sc.llm.requests), "không được gửi nội dung của chat id lạ cho LLM", None, "LLM đã nhận nội dung đó")
        sc.tg.push(CHAT, S_A)
        need(_wait(lambda: sc.by_name("My")), "sau đó tin từ chat được phép vẫn phải xử lý bình thường", None, clip(sc.people()))
        need(sc.by_name("Hùng") is None, "vẫn không có người từ chat lạ", None, clip(sc.people()))


@check(7, "Telegram: TELEGRAM_CHAT_ID trống thì không xử lý update nào")
def c7_nochat(root, n):
    with Scenario(root, n, tg=True, chat="") as sc:
        load(sc, (S_A, R_A))
        sc.tg.push(CHAT, S_A)
        time.sleep(2.0)
        need(not sc.people() and not sc.tg.sent, "chưa đặt TELEGRAM_CHAT_ID thì không được xử lý/ trả lời bất kỳ ai", f"update chat {CHAT}", f"people={clip(sc.people())} sent={clip(sc.tg.sent)}")


@check(7, "Telegram + LLM tắt: xếp hàng, báo đã xếp hàng, xả bù sau")
def c7_queue(root, n):
    with Scenario(root, n, tg=True) as sc:
        load(sc, (S_A, R_A))
        sc.llm.stop()
        sc.tg.push(CHAT, S_A)
        need(_wait(lambda: sc.tg.sent_to(CHAT), 8), "LLM tắt vẫn phải trả lời người dùng (xác nhận đã xếp hàng chờ)", None, clip(sc.tg.sent))
        need(sc.app.get("/api/queue").json.get("pending") == 1, "tin Telegram phải nằm trong hàng đợi", None, clip(sc.app.get("/api/queue").text))
        sc.llm.start()
        d = sc.app.post("/api/queue/drain", timeout=30)
        need(d.json.get("processed") == 1 and sc.by_name("My"), "LLM bật lại: xả bù phải lưu My", None, clip(d.text))


@check(7, "đẩy nhắc việc tới Telegram đúng một lần")
def c7_push(root, n):
    with Scenario(root, n, tg=True) as sc:
        load(sc, (S_BD, R_BD))
        sc.say(S_BD, at="2026-11-10T09:00:00")
        base = len(sc.tg.sent_to(CHAT))
        r = sc.app.post("/api/reminders/push?now=2026-12-17T09:00:00")
        need(r.status == 200 and r.json.get("sent", 0) >= 1, "POST /api/reminders/push phải gửi ít nhất 1 nhắc và trả {\"sent\": n}", "now=2026-12-17T09:00:00", f"{r.status} {clip(r.text)}")
        msgs = [m["text"] for m in sc.tg.sent_to(CHAT)[base:]]
        need(any("3 ngày" in m and "My" in m for m in msgs), "tin gửi qua Telegram phải chứa 'My' và '3 ngày'", None, clip(msgs))
        r2 = sc.app.post("/api/reminders/push?now=2026-12-17T09:00:00")
        need(r2.json.get("sent") == 0, "gọi lần hai không được gửi lại nhắc cũ (sent=0)", None, clip(r2.text))
        need(len(sc.tg.sent_to(CHAT)[base:]) == len(msgs), "Telegram không được nhận thêm tin ở lần hai", None, f"{len(sc.tg.sent_to(CHAT)) - base} so với {len(msgs)}")


@check(7, "riêng tư: mã không chứa URL lạ; token không lộ ra log")
def c7_privacy(root, n):
    allowed = {"127.0.0.1", "localhost", "api.telegram.org", "0.0.0.0"}
    bad = []
    for f in (root / "people_memory").rglob("*.py"):
        for m in re.finditer(r"https?://([A-Za-z0-9.\-]+)", f.read_text(errors="replace")):
            if m.group(1) not in allowed:
                bad.append(f"{f.name}: {m.group(0)}")
    need(not bad, "mã nguồn chỉ được chứa URL tới 127.0.0.1/localhost/api.telegram.org", "quét people_memory/*.py", "; ".join(bad[:5]))
    with Scenario(root, n, tg=True) as sc:
        load(sc, (S_A, R_A))
        sc.tg.push(CHAT, S_A)
        _wait(lambda: sc.by_name("My"))
        time.sleep(0.5)
        log = sc.app.logf.read_text(errors="replace")
        need(TOKEN not in log, "token Telegram không được xuất hiện trong log của ứng dụng", "đọc log", "log có chứa token")
        for p in (sc.app.data_dir).rglob("*"):
            if p.is_file() and p.suffix in (".db",):
                need(not db_contains(p, TOKEN), "token không được lưu vào DB", None, p.name)


# ---------------------------------------------------------------- chạy
def run(root: Path, stages: list[int], only: str | None) -> dict[int, list[tuple[str, str]]]:
    results: dict[int, list[tuple[str, str]]] = {s: [] for s in stages}
    n = 0
    for stage, name, fn in CHECKS:
        if stage not in stages or (only and only not in name):
            continue
        n += 1
        t0 = time.time()
        HINTS.clear()
        try:
            fn(root, n)
            results[stage].append((name, ""))
            print(f"✓ [S{stage}] {name}  ({time.time() - t0:.1f}s)", flush=True)
        except CheckFail as e:
            d = json.loads(str(e))
            lines = [f"✗ [S{stage}] {name}", f"   vấn đề: {d['what']}"]
            if d.get("sent") is not None:
                lines.append(f"   đã gửi: {clip(d['sent'])}")
            if d.get("got") is not None:
                lines.append(f"   nhận được: {clip(d['got'], 420)}")
            msg = "\n".join(lines)
            results[stage].append((name, msg))
            print(msg, flush=True)
        except Exception as e:  # lỗi bất ngờ trong oracle hoặc ứng dụng
            msg = f"✗ [S{stage}] {name}\n   lỗi bất ngờ khi chạy check: {type(e).__name__}: {clip(str(e), 300)}"
            results[stage].append((name, msg))
            print(msg, flush=True)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--stage", type=int)
    ap.add_argument("--upto", type=int)
    ap.add_argument("--only")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        print(json.dumps([{"id": k, "title": v[0], "brief": v[1]} for k, v in STAGES.items()], ensure_ascii=False))
        return 0
    root = Path(a.root).resolve()
    stages = [a.stage] if a.stage else list(range(1, (a.upto or 7) + 1))
    (root / ".oracle").mkdir(exist_ok=True)
    try:
        results = run(root, stages, a.only)
    finally:
        shutil.rmtree(root / ".oracle", ignore_errors=True)
    print()
    ok_all = True
    for s in stages:
        res = results[s]
        passed = sum(1 for _, m in res if not m)
        ok = bool(res) and passed == len(res)
        ok_all &= ok
        print(f"GIAI ĐOẠN {s} ({STAGES[s][0]}): {'ĐẠT' if ok else 'CHƯA'} ({passed}/{len(res)})")
    print("KẾT QUẢ:", "ĐẠT" if ok_all else "CHƯA ĐẠT")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

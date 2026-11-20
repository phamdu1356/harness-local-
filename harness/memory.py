"""Bộ nhớ ngoài cho agent: SQLite FTS5 (BM25), không cần model embedding.

Lưu: lượt hội thoại ("turn"), kết quả tool ("tool"), ghi chú do model/người dùng chủ động ("note").
Tìm kiếm không phân biệt dấu tiếng Việt. Mục còn nằm trong context hiện tại bị loại khỏi kết quả tìm tự động.
"""
from __future__ import annotations

import array
import hashlib
import re
import sqlite3
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .embed import EmbedError, cosine
from .tools import P, Tool, clip

CHUNK = 1200
VEC_MIN = 0.5  # bge-m3: mục liên quan ≥ 0.55, không liên quan ≤ 0.47 (đo trên máy)
STOP = set("là và của có không cho được này đó nay hôm tôi bạn gì nào thì mà ở trong với một các những đang đã sẽ cũng như để khi "
           "the a an of to is in and or".split())


def fold(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", t.lower().replace("đ", "d")) if unicodedata.category(c) != "Mn")


STOP_FOLDED = {fold(w) for w in STOP}


def lexically_strong(content: str, query: str) -> bool:
    """Có từ khoá (≥3 ký tự, không phải từ phổ biến) của câu hỏi xuất hiện nguyên vẹn trong nội dung.
    Câu hỏi gõ không dấu thì so sánh sau khi bỏ dấu."""
    q_low = query.lower()
    unaccented = q_low == fold(q_low)
    norm = fold if unaccented else str.lower
    stop = STOP_FOLDED if unaccented else STOP
    words = set(re.findall(r"\w+", norm(content)))
    return any(len(w) >= 3 and w not in stop and w in words for w in re.findall(r"\w+", norm(query)))


@dataclass
class Hit:
    content: str
    kind: str
    ts: float
    session: str
    turn: int
    score: float


def ago(ts: float) -> str:
    d = time.time() - ts
    for unit, sec in (("ngày", 86400), ("giờ", 3600), ("phút", 60)):
        if d >= sec:
            return f"{int(d // sec)} {unit} trước"
    return "vừa xong"


def _chunks(text: str, size: int = CHUNK) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    out, cur = [], ""
    for line in text.splitlines():
        while len(line) > size:  # dòng quá dài
            if cur:
                out.append(cur)
                cur = ""
            out.append(line[:size])
            line = line[size:]
        if len(cur) + len(line) + 1 > size:
            out.append(cur)
            cur = ""
        cur += line + "\n"
    if cur.strip():
        out.append(cur)
    return [c.strip() for c in out if c.strip()]


_MEM_DDL = (
    'CREATE VIRTUAL TABLE IF NOT EXISTS mem USING fts5('
    'content UNINDEXED, folded, kind UNINDEXED, session UNINDEXED, turn UNINDEXED, ts UNINDEXED, '
    'tokenize="unicode61 remove_diacritics 2")'
)


class Memory:
    def __init__(self, path: str | Path = ":memory:", session: str = "s0", embedder=None):
        self.session, self.embedder = session, embedder
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            gi = Path(path).parent / ".gitignore"
            if not gi.exists():
                gi.write_text("*\n")
        self.db = sqlite3.connect(str(path))
        cols = [r[1] for r in self.db.execute("PRAGMA table_info(mem)").fetchall()]
        if cols and "folded" not in cols:
            old = self.db.execute("SELECT rowid, content, kind, session, turn, ts FROM mem").fetchall()
            self.db.execute("DROP TABLE mem")
            self.db.execute(_MEM_DDL)
            self.db.executemany(
                "INSERT INTO mem(rowid, content, folded, kind, session, turn, ts) VALUES (?,?,?,?,?,?,?)",
                [(rid, c, fold(c), k, s, t, ts) for rid, c, k, s, t, ts in old],
            )
        else:
            self.db.execute(_MEM_DDL)
        self.db.execute("CREATE TABLE IF NOT EXISTS seen(h TEXT PRIMARY KEY)")
        self.db.execute("CREATE TABLE IF NOT EXISTS vec(rid INTEGER PRIMARY KEY, v BLOB)")
        self.db.commit()

    def close(self) -> None:
        if getattr(self, "db", None) is not None:
            self.db.close()
            self.db = None

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def add(self, content: str, kind: str = "note", turn: int = 0) -> int:
        new: list[tuple[int, str]] = []
        for part in _chunks(content):
            h = hashlib.sha1(f"{kind}|{part}".encode()).hexdigest()
            if self.db.execute("SELECT 1 FROM seen WHERE h=?", (h,)).fetchone():
                continue
            self.db.execute("INSERT INTO seen VALUES (?)", (h,))
            cur = self.db.execute("INSERT INTO mem(content, folded, kind, session, turn, ts) VALUES (?,?,?,?,?,?)",
                                  (part, fold(part), kind, self.session, turn, time.time()))
            new.append((cur.lastrowid, part))
        self.db.commit()
        self._embed_rows(new)
        return len(new)

    def _embed_rows(self, rows: list[tuple[int, str]]) -> int:
        """Embedding lỗi thì bỏ qua: mục đó vẫn tìm được bằng BM25, và được bù vector ở lần backfill sau."""
        if not self.embedder or not rows:
            return 0
        try:
            vecs = self.embedder.embed([t for _, t in rows])
        except EmbedError:
            return 0
        pairs = [(rid, v.tobytes()) for (rid, _), v in zip(rows, vecs)]
        self.db.executemany("INSERT OR REPLACE INTO vec VALUES (?,?)", pairs)
        self.db.commit()
        return len(pairs)

    def backfill(self, limit: int = 300) -> int:
        """Nhúng các mục chưa có vector (ghi lúc chưa bật embedding). Trả về số mục đã bù."""
        if not self.embedder:
            return 0
        rows = self.db.execute("SELECT rowid, content FROM mem WHERE rowid NOT IN (SELECT rid FROM vec) "
                               "ORDER BY rowid DESC LIMIT ?", (limit,)).fetchall()
        total = 0
        for i in range(0, len(rows), 16):
            total += self._embed_rows(rows[i:i + 16])
        return total

    @staticmethod
    def _match_query(q: str) -> str:
        q_low = q.lower()
        stop = STOP_FOLDED if q_low == fold(q_low) else STOP
        words = sorted({fold(w) for w in re.findall(r"\w{2,}", q_low) if w not in stop}, key=len, reverse=True)[:12]
        return " OR ".join(f'"{w}"' for w in words)

    def _vector_rank(self, query: str, limit: int, first_live_turn: int | None = None,
                     kinds: tuple[str, ...] | None = None) -> list[tuple[int, float]] | None:
        """[(rowid, cosine)] giảm dần; None nếu embedding không dùng được."""
        if not self.embedder:
            return None
        try:
            qv = self.embedder.embed([query])[0]
        except EmbedError:
            return None
        conds, params = [], []
        if kinds:
            conds.append(f"m.kind IN ({','.join('?' * len(kinds))})")
            params.extend(kinds)
        if first_live_turn is not None:
            conds.append("NOT (m.session = ? AND CAST(m.turn AS INTEGER) >= ?)")
            params.extend([self.session, first_live_turn])
        where = (" WHERE " + " AND ".join(conds)) if conds else ""
        sql = f"SELECT v.rid, v.v FROM vec v JOIN mem m ON m.rowid = v.rid{where}"
        scored = [(rid, cosine(qv, array.array("f", blob))) for rid, blob in self.db.execute(sql, params)]
        scored = sorted((x for x in scored if x[1] >= VEC_MIN), key=lambda x: -x[1])
        return scored[:limit]

    def search(self, query: str, k: int = 4, first_live_turn: int | None = None,
               kinds: tuple[str, ...] | None = None) -> list[Hit]:
        """Tìm lai: BM25 (khớp từ) + vector (khớp nghĩa) hợp nhất bằng reciprocal rank fusion.
        Khi có embedding, mục chỉ khớp BM25 phải chứa từ khoá nguyên vẹn mới được nhận (chống nhiễu do bỏ dấu).
        first_live_turn: bỏ qua mục của phiên hiện tại từ lượt này trở đi (đang nằm trong context)."""
        mq = self._match_query(query)
        pool = k * 6 + 10
        bm25: list[int] = []
        if mq:
            conds, params = ["mem MATCH ?"], [mq]
            if kinds:
                conds.append(f"kind IN ({','.join('?' * len(kinds))})")
                params.extend(kinds)
            if first_live_turn is not None:
                conds.append("NOT (session = ? AND CAST(turn AS INTEGER) >= ?)")
                params.extend([self.session, first_live_turn])
            sql = f"SELECT rowid FROM mem WHERE {' AND '.join(conds)} ORDER BY bm25(mem) LIMIT ?"
            params.append(pool)
            bm25 = [r for (r,) in self.db.execute(sql, params).fetchall()]
        vec = self._vector_rank(query, pool, first_live_turn=first_live_turn, kinds=kinds)
        vec_ids = {rid for rid, _ in vec or []}
        ranks: dict[int, float] = {}
        for i, rid in enumerate(bm25):
            if vec is not None and rid not in vec_ids:
                row = self.db.execute("SELECT content FROM mem WHERE rowid=?", (rid,)).fetchone()
                if not row or not lexically_strong(row[0], query):
                    continue
            ranks[rid] = ranks.get(rid, 0) + 1 / (60 + i)
        for i, (rid, _) in enumerate(vec or []):
            ranks[rid] = ranks.get(rid, 0) + 1 / (60 + i)
        hits = []
        for rid, score in sorted(ranks.items(), key=lambda x: -x[1]):
            row = self.db.execute("SELECT content, kind, ts, session, turn FROM mem WHERE rowid=?", (rid,)).fetchone()
            if not row:
                continue
            content, kind, ts, session, turn = row
            if kinds and kind not in kinds:
                continue
            if first_live_turn is not None and session == self.session and int(turn) >= first_live_turn:
                continue
            hits.append(Hit(content, kind, ts, session, int(turn), -(score + (0.02 if kind == "note" else 0.0))))
        hits.sort(key=lambda h: h.score)
        return hits[:k]

    def stats(self) -> dict[str, int]:
        st = dict(self.db.execute("SELECT kind, COUNT(*) FROM mem GROUP BY kind").fetchall())
        if self.embedder:
            st["vector"] = self.db.execute("SELECT COUNT(*) FROM vec").fetchone()[0]
        return st

    def clear(self) -> None:
        self.db.execute("DELETE FROM mem")
        self.db.execute("DELETE FROM seen")
        self.db.execute("DELETE FROM vec")
        self.db.commit()


def format_hits(hits: list[Hit], limit: int = 2400) -> str:
    out, used = [], 0
    for h in hits:
        item = f"[{h.kind} · {ago(h.ts)}] {clip(h.content, 700)}"
        if used + len(item) > limit:
            break
        out.append(item)
        used += len(item)
    return "\n---\n".join(out)


def memory_tools(mem: Memory, live_turn: "callable") -> dict[str, Tool]:
    def remember(text: str) -> str:
        n = mem.add(text, "note", live_turn())
        return "Đã lưu vào bộ nhớ." if n else "Ghi chú này đã có sẵn."

    def recall(query: str, k: int = 5) -> str:
        hits = mem.search(query, k=max(1, min(k, 10)))
        return format_hits(hits, 6000) or "Không tìm thấy gì liên quan trong bộ nhớ."

    tools = [
        Tool("remember", "Lưu một ghi chú lâu dài vào bộ nhớ ngoài (sở thích người dùng, quyết định, sự thật về dự án).",
             {"text": P("string", "nội dung ghi chú, tự đủ nghĩa")}, remember, read_only=True),
        Tool("recall", "Tìm trong bộ nhớ ngoài: các lượt hội thoại cũ, kết quả tool cũ đã bị lược khỏi context, ghi chú.",
             {"query": P("string", "từ khoá cần tìm"), "k": P("integer", "số kết quả, mặc định 5", True)},
             recall, read_only=True),
    ]
    return {t.name: t for t in tools}

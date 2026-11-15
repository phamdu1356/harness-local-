"""Chế độ --stages: làm theo từng giai đoạn nghiệm thu, kiểm chứng bằng một oracle độc lập (ngoài sandbox của agent).

Oracle là một script do người vận hành viết và đặt NGOÀI thư mục làm việc của agent, nên agent không đọc/sửa được; nó chạy app thật
(trong sandbox) rồi báo cụ thể chỗ sai. Giao ước dòng lệnh:
    oracle.py --list                  -> JSON [{"id", "title", "brief"}]
    oracle.py --root DIR --upto K     -> chạy giai đoạn 1..K, in dòng "✗ ..." (kèm chi tiết thụt vào) cho mỗi lỗi; mã thoát 0 nếu đạt
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .tools import clip
from .ui import UI

STAGE_PROMPT = """GIAI ĐOẠN {id}/{n}: {title}

{brief}

Hệ thống nghiệm thu độc lập đã chạy ứng dụng của bạn thật và đây là kết quả hiện tại (giai đoạn {id} và mọi giai đoạn trước đều phải đạt):
{output}

Hãy sửa/viết mã để các lỗi trên hết. Đọc thông báo lỗi thật (dòng "vấn đề", "đã gửi", "nhận được") thay vì đoán; đọc mã liên quan bằng
read_file trước khi sửa; mỗi tính năng có test riêng trong tests/ (test dùng thư mục tạm và tự dọn). Hệ thống sẽ tự chạy lại kiểm tra
khi bạn báo xong, và bạn chỉ xong khi nó đạt. Chỉ làm giai đoạn này."""

FOLLOW_PROMPT = """GIAI ĐOẠN {id}/{n} ({title}) vẫn CHƯA đạt. Kết quả nghiệm thu mới nhất:
{output}

Sửa tiếp đúng các lỗi này (đọc mã liên quan trước), rồi báo xong khi bạn đã tự thấy nó chạy đúng."""


@dataclass
class StageResult:
    id: int
    title: str
    ok: bool
    rounds: int
    output: str


class Oracle:
    def __init__(self, script: str | Path, root: str | Path, timeout: int = 900):
        self.script, self.root, self.timeout = str(script), str(root), timeout

    def list_stages(self) -> list[dict]:
        r = subprocess.run([sys.executable, self.script, "--list"], capture_output=True, text=True, timeout=60)
        return json.loads(r.stdout)

    def run(self, upto: int) -> tuple[bool, str]:
        try:
            r = subprocess.run([sys.executable, self.script, "--root", self.root, "--upto", str(upto)],
                               capture_output=True, text=True, timeout=self.timeout, cwd=self.root)
        except subprocess.TimeoutExpired:
            return False, f"✗ oracle chạy quá {self.timeout}s (ứng dụng có thể bị treo)"
        return r.returncode == 0, condense(r.stdout + (("\n" + r.stderr[-500:]) if r.returncode and r.stderr.strip() else ""))


def condense(out: str) -> str:
    """Giữ các lỗi (✗ + dòng chi tiết thụt vào) và dòng tổng kết; bỏ các dòng ✓; gộp các lỗi có chi tiết y hệt nhau."""
    blocks: list[list[str]] = []
    tail: list[str] = []
    for line in out.splitlines():
        if line.startswith("✗"):
            blocks.append([line])
        elif line.startswith("   ") and blocks and not tail:
            blocks[-1].append(line)
        elif line.startswith(("GIAI ĐOẠN", "KẾT QUẢ")):
            tail.append(line)
    seen: dict[tuple, int] = {}
    kept: list[list[str]] = []
    for blk in blocks:
        key = tuple(blk[1:])
        if key and key in seen:
            kept[seen[key]].append("   (check khác cũng bị đúng lỗi này: " + blk[0][2:60] + "…)")
            continue
        seen[key] = len(kept)
        kept.append(blk)
    lines = [l for blk in kept for l in blk] + tail
    return "\n".join(lines) if lines else out.strip()


def run_stages(agent, oracle: Oracle, step_budget: int = 40, max_rounds: int = 4, start: int = 1,
               unit_verify=None, ui: UI | None = None, on_stage_done=None) -> list[StageResult]:
    ui = ui or agent.ui
    stages = [s for s in oracle.list_stages() if s["id"] >= start]
    n_total = max((s["id"] for s in oracle.list_stages()), default=len(stages))
    results: list[StageResult] = []
    saved_steps, saved_verify = agent.max_steps, agent.verify
    agent.max_steps = step_budget
    try:
        for st in stages:
            def gate(upto=st["id"]):
                ok, out = oracle.run(upto)
                if ok and unit_verify:
                    ok_u, out_u = unit_verify()
                    if not ok_u:
                        return False, "Bộ test của chính bạn (tests/) đang thất bại:\n" + out_u
                return ok, out
            agent.verify = gate
            ui.w(ui.c("accent", f"\n▶ Giai đoạn {st['id']}/{n_total}: {st['title']}"))
            ok, out = gate()
            rounds = 0
            while not ok and rounds < max_rounds:
                rounds += 1
                tpl = STAGE_PROMPT if rounds == 1 else FOLLOW_PROMPT
                agent.run(tpl.format(id=st["id"], n=n_total, title=st["title"], brief=st["brief"], output=clip(out, 3500)))
                ok, out = gate()
                ui.w(ui.c("ok" if ok else "err", f"{'✓' if ok else '✗'} giai đoạn {st['id']} sau vòng {rounds}: {'ĐẠT' if ok else 'chưa đạt'}"))
            results.append(StageResult(st["id"], st["title"], ok, rounds, out))
            if on_stage_done:
                on_stage_done()
            if not ok:   # các giai đoạn sau phụ thuộc giai đoạn này
                ui.error(f"giai đoạn {st['id']} chưa đạt sau {max_rounds} vòng: dừng")
                break
    finally:
        agent.max_steps, agent.verify = saved_steps, saved_verify
    return results


def exit_code(results: list[StageResult], last_stage_id: int) -> int:
    """0 chỉ khi đã chạy tới giai đoạn CUỐI CÙNG của oracle và mọi giai đoạn đã chạy đều đạt; ngược lại 2."""
    return 0 if results and last_stage_id > 0 and all(r.ok for r in results) and results[-1].id >= last_stage_id else 2


def report(results: list[StageResult]) -> str:
    if not results:
        return "Không có giai đoạn nào được chạy."
    lines = [f"Đạt {sum(r.ok for r in results)}/{len(results)} giai đoạn đã chạy:"]
    lines += [f"  {'✓' if r.ok else '✗'} {r.id}. {r.title} ({r.rounds} vòng)" for r in results]
    return "\n".join(lines)

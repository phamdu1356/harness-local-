"""Chế độ --goal: biến một mục tiêu lớn thành chuỗi việc nhỏ, mỗi việc có ngân sách bước riêng.

Vì sao cần: giao cả mục tiêu lớn trong một lượt, model nhỏ hay dồn toàn bộ bước vào một chỗ khó và bỏ quên phần còn lại.
Cách làm: (1) agent lập kế hoạch bằng tool todo (chỉ được đọc, không được sửa gì); (2) harness NẮM kế hoạch đó và giao
TỪNG việc, hết ngân sách bước thì đánh dấu bị chặn rồi chuyển việc kế tiếp, model có viết lại todo cũng không làm lệch
kế hoạch; (3) cổng --verify (nếu có) chạy sau mỗi việc, chỉ chặn lỗi MỚI so với trước việc đó.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from .agent import _no_regression, fingerprint
from .tools import clip
from .ui import UI

ITEM_RX = re.compile(r"^\s*\[( |~|x|X|!)\]\s*(.+?)\s*$")
PLAN_TOOLS = ("read_file", "list_dir", "glob", "grep", "todo", "recall")   # giai đoạn lập kế hoạch: chỉ đọc

PLAN_PROMPT = """MỤC TIÊU LỚN:
{goal}

Bước này CHỈ LẬP KẾ HOẠCH (bạn chỉ có tool đọc và todo, chưa sửa được gì). Hãy khảo sát nhanh thư mục hiện tại (xem file
đã có, đọc yêu cầu) rồi gọi tool todo ghi kế hoạch gồm {lo}–{hi} việc NHỎ, mỗi dòng dạng `[ ] mô tả`. Mỗi việc phải nhỏ
đến mức làm xong và kiểm chứng được trong khoảng {half} bước (một hàm/một module nhỏ + test, hoặc một lỗi cụ thể); việc lớn
thì tách ra. Không lặp lại cùng một việc. Xếp theo ưu tiên của yêu cầu, việc nền tảng trước. Đừng đưa việc đã làm xong vào
kế hoạch. Sau khi gọi todo hãy dừng."""

ITEM_PROMPT = """VIỆC HIỆN TẠI ({i}/{n}): {item}

Chỉ làm đúng việc này, tối đa {budget} bước. Đọc mã liên quan bằng read_file trước khi sửa; viết test cho phần mới và chạy
test. Khi xong, gọi tool todo đánh `[x]` cho việc này rồi báo ngắn gọn bạn đã làm gì và test nào chứng minh. Nếu thấy việc quá
lớn, làm phần cốt lõi trước. Đừng sa vào việc khác ngoài danh sách. Kế hoạch tổng do hệ thống quản lý, đừng viết lại nó."""


CORRECT_PROMPT = """VIỆC HIỆN TẠI ({i}/{n}): {item}

Hệ thống kiểm tra độc lập và thấy việc này CHƯA đạt: {problem}
Hệ thống chỉ tính là xong khi có thay đổi thật trên file kèm test chứng minh hành vi, và toàn bộ test đang pass. Hãy làm đúng
phần còn thiếu: viết mã/test bằng write_file hoặc edit_file, chạy test cho đến khi pass, rồi báo ngắn gọn file nào đã đổi và test
nào chứng minh. Nếu chức năng thực ra đã có sẵn từ trước, hãy viết MỘT test chứng minh nó chạy đúng (test mới là thay đổi thật).
Còn {left} lần bổ sung."""


REPAIR_PROMPT = """Việc trước vừa HẾT BƯỚC và để lại bộ test ĐỎ (có lỗi MỚI so với lúc bắt đầu):
{output}

Chưa được sang việc khác khi bộ test còn đỏ. Hãy đưa nó về xanh ngay: đọc thông báo lỗi và đoạn mã/test liên quan bằng read_file,
tìm nguyên nhân thật (có thể chính test viết sai: kỳ vọng sai, cách gọi assert sai, dữ liệu phụ thuộc thời gian), rồi sửa bằng
edit_file. Nếu phần làm dở của việc trước là nguyên nhân, hãy sửa cho xong hoặc gỡ phần dở đó đi. Đừng thêm tính năng mới."""


def _is_test_file(c: str) -> bool:
    p = Path(c)
    if p.suffix != ".py":
        return False
    name = p.name.lower()
    return (name.startswith("test_") or name.endswith("_test.py")
            or name in {"test.py", "tests.py", "conftest.py"}
            or any(part.lower() in {"test", "tests"} for part in p.parts[:-1]))


def completion_problem(changed: list[str], require_change: bool, require_tests: bool) -> str:
    """Vấn đề khiến việc chưa được tính là xong (rỗng nếu ổn): không đổi gì, hoặc đổi mã mà không có test đi kèm."""
    if require_change and not changed:
        return "bạn báo xong nhưng CHƯA thay đổi file nào."
    code = [c for c in changed if Path(c).suffix == ".py" and not _is_test_file(c)]
    tests = [c for c in changed if _is_test_file(c)]
    if require_tests and code and not tests:
        return f"bạn đã đổi mã ({', '.join(code[:3])}) nhưng chưa thêm/sửa test nào chứng minh hành vi mới."
    return ""


@dataclass
class Item:
    text: str
    status: str = "pending"  # pending | done | blocked
    note: str = ""


@dataclass
class ItemResult:
    index: int
    text: str
    status: str  # done | blocked
    note: str = ""


class GoalResults(list):
    def __init__(self, items=(), final_ok: bool = True):
        super().__init__(items)
        self.final_ok = final_ok


def parse_plan(lines: list[str]) -> list[tuple[str, str]]:
    out = []
    for line in lines:
        m = ITEM_RX.match(line)
        if m:
            out.append((m.group(1).lower(), m.group(2)))
    return out


def norm(text: str) -> str:
    return re.sub(r"\W+", " ", re.sub(r"\(bị chặn.*?\)", "", text.lower())).strip()


def build_items(todo: list[str], max_items: int) -> list[Item]:
    """Kế hoạch ban đầu của agent -> danh sách việc; bỏ việc trùng và việc đã đánh xong."""
    items, seen = [], set()
    for st, text in parse_plan(todo):
        key = norm(text)
        if st == "x" or not key or key in seen:
            continue
        seen.add(key)
        items.append(Item(text))
    return items[:max_items]


def render(items: list[Item], current: int | None = None) -> list[str]:
    out = []
    for i, it in enumerate(items):
        mark = {"done": "x", "blocked": "!"}.get(it.status, "~" if i == current else " ")
        out.append(f"[{mark}] {it.text}" + (f" ({it.note})" if it.note else ""))
    return out


def _model_marked_done(todo: list[str], item: Item) -> bool:
    key = norm(item.text)
    if not key:
        return False
    for st, t in parse_plan(todo):
        if st != "x":
            continue
        nt = norm(t)
        if nt and (nt == key or nt.startswith(key[:40]) or key.startswith(nt[:40])):
            return True
    return False


SKIP_DIRS = {".harness", "__pycache__", ".git", "node_modules", ".venv", "venv", "dist", "build", ".mypy_cache"}
SKIP_SUFFIX = {".pyc"}
DATA_SUFFIX = {".db", ".sqlite", ".sqlite3", ".log"}   # test/chạy thử sinh ra chúng: tạo/sửa không tính, nhưng XOÁ thì tính (dọn rác)


def snapshot(root: Path) -> dict[str, tuple[int, int]]:
    """Dấu vết các file của dự án (kích thước, thời điểm sửa) để biết một việc có thật sự làm thay đổi gì không."""
    snap = {}
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            p = Path(dirpath) / f
            if p.suffix in SKIP_SUFFIX:
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            snap[str(p.relative_to(root))] = (st.st_size, st.st_mtime_ns)
    return snap


def changed_files(before: dict, after: dict) -> list[str]:
    def noise(p: str) -> bool:
        return Path(p).suffix in DATA_SUFFIX
    changed = {p for p in after if before.get(p) != after[p] and not noise(p)}
    return sorted(changed | {p for p in before if p not in after})


def _dump_goal_state(items: list[Item], result_notes: list[str], baseline, goal: str = "") -> dict:
    if isinstance(baseline, (set, frozenset)):
        bl_val = sorted(baseline)
    elif isinstance(baseline, str):
        bl_val = baseline
    else:
        bl_val = None
    return {
        "goal": goal,
        "items": [{"text": it.text, "status": it.status, "note": it.note} for it in items],
        "result_notes": list(result_notes),
        "baseline": bl_val,
    }


def _restore_goal_state(agent, todo: list[str], max_items: int, goal: str = "") -> tuple[list[Item], list[str], frozenset[str] | str | None, bool] | None:
    gs = getattr(agent, "goal_state", None)
    if isinstance(gs, dict) and isinstance(gs.get("items"), list) and gs["items"]:
        if gs.get("goal") and goal and gs["goal"] != goal:
            return None
        items: list[Item] = []
        for raw in gs["items"][:max_items]:
            if not isinstance(raw, dict) or not raw.get("text"):
                continue
            st = raw.get("status", "pending")
            items.append(Item(str(raw["text"]), st if st in ("pending", "done", "blocked") else "pending", str(raw.get("note", ""))))
        if items:
            raw_notes = gs.get("result_notes") if isinstance(gs.get("result_notes"), list) else []
            notes = [str(raw_notes[i]) if i < len(raw_notes) else it.note for i, it in enumerate(items)]
            raw_bl = gs.get("baseline")
            if isinstance(raw_bl, list):
                bl = frozenset(str(x) for x in raw_bl)
            elif isinstance(raw_bl, str):
                bl = raw_bl
            else:
                bl = None
            return items, notes, bl, ("baseline" in gs)
    parsed = parse_plan(todo)
    if not parsed:
        return None
    items, notes = [], []
    for st, text in parsed[:max_items]:
        status = {"x": "done", "!": "blocked"}.get(st, "pending")
        note = ""
        if status == "blocked":
            m = re.match(r"^(.*?)\s+\((bị chặn:.*)\)$", text)
            if m:
                text, note = m.group(1), m.group(2)
        items.append(Item(text, status, note))
        notes.append(note)
    return items, notes, None, False


def run_goal(agent, goal: str, step_budget: int = 30, max_items: int = 10, plan_steps: int = 15,
             ui: UI | None = None, require_change: bool = True, require_tests: bool = True,
             fix_rounds: int = 2, on_item_done=None) -> GoalResults:
    ui = ui or agent.ui
    todo = agent.tools["todo"].state
    saved_steps, saved_tools, saved_verify = agent.max_steps, agent.tools, agent.verify
    results: list[ItemResult] = []
    resumed = bool(getattr(agent, "_resumed", False))
    if hasattr(agent, "_resumed"):
        agent._resumed = False
    try:
        restored = _restore_goal_state(agent, todo, max_items, goal) if resumed else None
        if restored:
            items, result_notes, baseline, has_baseline = restored
            ui.w(ui.c("muted", f"◌ tiếp tục kế hoạch đã lưu ({len(items)} việc)"))
            if not has_baseline and agent.verify:
                ok0, out0 = agent.verify()
                if not ok0:
                    baseline = fingerprint(out0)
                    ui.w(ui.c("muted", "◌ đã có lỗi kiểm tra từ trước khi tiếp tục: chỉ chặn lỗi MỚI so với mốc này"))
        else:
            todo.clear()
            agent.goal_state = None
            agent.max_steps = plan_steps
            agent.tools = {n: t for n, t in saved_tools.items() if n in PLAN_TOOLS}   # chỉ đọc khi lập kế hoạch
            agent.verify = None   # lập kế hoạch thì chưa có gì để kiểm tra (tránh bị kéo đi sửa test ngay lúc này)
            ui.w(ui.c("accent", "▶ Giai đoạn 1: lập kế hoạch (chỉ đọc)"))
            agent.run(PLAN_PROMPT.format(goal=goal, lo=4, hi=max(5, max_items), half=max(5, step_budget // 2)))
            agent.tools, agent.verify = saved_tools, saved_verify
            items = build_items(todo, max_items)
            if not items:
                ui.error("agent không lập được kế hoạch (todo trống)")
                return GoalResults(results, final_ok=False)
            result_notes = [""] * len(items)
            ui.w(ui.c("muted", f"◌ kế hoạch gồm {len(items)} việc (do hệ thống nắm giữ, model viết lại todo cũng không đổi)"))
            baseline = None   # mốc lỗi CỐ ĐỊNH từ đầu: lỗi do một việc trước gây ra không được coi là "có sẵn"
            if agent.verify:
                ok0, out0 = agent.verify()
                if not ok0:
                    baseline = fingerprint(out0)
                    ui.w(ui.c("muted", "◌ đã có lỗi kiểm tra từ trước khi bắt đầu: chỉ chặn lỗi MỚI so với mốc này"))
            agent.goal_state = _dump_goal_state(items, result_notes, baseline, goal)
        agent.max_steps = step_budget
        for idx, item in enumerate(items):
            if item.status != "pending":
                results.append(ItemResult(idx + 1, item.text, item.status, result_notes[idx]))
                continue
            todo[:] = render(items, idx)
            ui.w(ui.c("accent", f"\n▶ Việc {idx + 1}/{len(items)}: {item.text}"))
            agent.verify_baseline = baseline
            before = snapshot(agent.ws.root)
            prompt, attempts = ITEM_PROMPT.format(i=idx + 1, n=len(items), item=item.text, budget=step_budget), 0
            while True:
                agent.run(prompt)
                changed = changed_files(before, snapshot(agent.ws.root))
                problem = completion_problem(changed, require_change, require_tests)
                if agent.exhausted or not problem or attempts >= fix_rounds:
                    break
                attempts += 1   # cho cơ hội bổ sung có hướng dẫn cụ thể thay vì chặn ngay
                ui.w(ui.c("muted", f"◌ chưa đạt: {problem} → yêu cầu bổ sung ({attempts}/{fix_rounds})"))
                prompt = CORRECT_PROMPT.format(i=idx + 1, n=len(items), item=item.text, problem=problem,
                                               left=fix_rounds - attempts + 1)
            if problem and not agent.exhausted:
                item.status, item.note = "blocked", f"bị chặn: {problem}"
                result_notes[idx] = problem
                results.append(ItemResult(idx + 1, item.text, "blocked", problem))
                ui.w(ui.c("err", f"✗ việc {idx + 1} chưa đạt sau {attempts} lần bổ sung: {problem}"))
            elif agent.verify and not agent.exhausted and getattr(agent, "last_verify_ok", None) is False:
                item.status, item.note = "blocked", "bị chặn: kiểm tra không đạt"
                result_notes[idx] = "kiểm tra không đạt"
                results.append(ItemResult(idx + 1, item.text, "blocked", "kiểm tra không đạt"))
                ui.w(ui.c("err", f"✗ việc {idx + 1} không đạt kiểm tra sau các vòng sửa"))
            elif (not agent.verify and _model_marked_done(todo, item)) or not agent.exhausted:   # tự đánh [x] hoặc kết thúc tự nhiên
                item.status = "done"
                result_notes[idx] = "đã đổi: " + ", ".join(changed[:4])
                results.append(ItemResult(idx + 1, item.text, "done", result_notes[idx]))
                ui.w(ui.c("ok", f"✓ xong việc {idx + 1} (đổi {len(changed)} file)"))
            else:
                item.status, item.note = "blocked", f"bị chặn: hết {step_budget} bước"
                result_notes[idx] = f"hết {step_budget} bước"
                results.append(ItemResult(idx + 1, item.text, "blocked", result_notes[idx]))
                ui.w(ui.c("err", f"✗ việc {idx + 1} bị chặn (hết {step_budget} bước), chuyển việc kế tiếp"))
            if agent.verify and agent.exhausted:   # hết bước thì không có cổng kiểm tra nào chạy: tự kiểm tra, không để suite đỏ lan sang việc sau
                ok_r, out_r = agent.verify()
                if not ok_r and not _no_regression(fingerprint(out_r), baseline):
                    ui.w(ui.c("warn", "◌ hết bước và để lại bộ test ĐỎ → phải đưa về xanh trước khi sang việc kế tiếp"))
                    agent.run(REPAIR_PROMPT.format(output=clip(out_r, 2500)))
                    results[-1].note += " (đã để lại test đỏ, đã yêu cầu sửa)"
                    result_notes[idx] = results[-1].note
            todo[:] = render(items)   # khôi phục kế hoạch chuẩn, bỏ mọi chỉnh sửa của model
            agent.goal_state = _dump_goal_state(items, result_notes, baseline, goal)
            if on_item_done:
                on_item_done()   # lưu phiên sau mỗi việc: server chết giữa chừng thì không mất cả phiên
    finally:
        agent.max_steps, agent.tools, agent.verify = saved_steps, saved_tools, saved_verify
        agent.verify_baseline = None
    out_results = GoalResults(results)
    if agent.verify and out_results:   # kiểm tra cuối, KHÔNG nới theo mốc: toàn bộ kiểm tra phải đạt
        ok, out = agent.verify()
        out_results.final_ok = ok
        tail = out.strip().splitlines()[-1] if out.strip() else ""
        ui.w(ui.c("ok" if ok else "err", f"\n{'✓' if ok else '✗'} KIỂM TRA CUỐI CÙNG (không nới theo mốc): {'đạt' if ok else 'KHÔNG đạt'} — {tail[:100]}"))
    return out_results


def report(results: list[ItemResult]) -> str:
    if not results:
        return "Không có việc nào được thực hiện."
    done = sum(r.status == "done" for r in results)
    lines = [f"Hoàn thành {done}/{len(results)} việc:"]
    lines += [f"  {'✓' if r.status == 'done' else '✗'} {r.index}. {r.text}" + (f" — {r.note}" if r.note else "")
              for r in results]
    return "\n".join(lines)

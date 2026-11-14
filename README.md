# harness

Agent coding CLI nhỏ (Python, không dependency) cho LLM local chạy qua **LM Studio**.

## Dùng
1. LM Studio → load model → tab **Developer** → **Start Server** (mặc định `localhost:1234`).
2. Chạy:
```bash
python3 -m harness            # REPL trong thư mục hiện tại
python3 -m harness -p "giải thích main.py" -C ~/code/dự-án
python3 -m harness -y         # tự duyệt write/edit/bash
python3 -m harness --text-tools   # nếu model không hỗ trợ tool-calling native
```
Tuỳ chọn: `-m <model-id>`, `--host`, `--temperature`, `--max-steps`; env `HARNESS_MODEL`, `LMSTUDIO_HOST`.

## Cấu trúc
- `harness/client.py` — client OpenAI-compatible, streaming, gom tool_calls
- `harness/tools.py` — read_file, list_dir, grep, write_file, edit_file, bash (giới hạn trong workspace)
- `harness/agent.py` — vòng lặp agent, duyệt quyền, fallback `<tool_call>` dạng text
- `tests/` — `python3 -m unittest discover -s tests`

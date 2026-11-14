# harness

Agent coding CLI nhỏ (Python, không dependency) cho LLM local.
Chạy qua **LM Studio** hoặc tự chạy **llama-server** (llama.cpp) không cần mở giao diện.

## Dùng
```bash
python3 -m harness                    # REPL, dùng server LM Studio (localhost:1234)
python3 -m harness --local            # tự chạy llama-server với model .gguf nhỏ nhất trong ~/.lmstudio/models
python3 -m harness --local DeepSeek   # chọn model theo một phần tên (hoặc đường dẫn .gguf)
python3 -m harness --local DeepSeek --cpu --think off   # chạy hoàn toàn bằng CPU, không dùng VRAM
python3 -m harness -p "giải thích main.py" -C ~/code/dự-án
python3 -m harness -y                 # tự duyệt thao tác ghi/xoá/bash/web
```
Tuỳ chọn: `-m`, `--host`, `--temperature`, `--max-steps`, `--system "..."|@file`, `--text-tools`,
`--cpu --threads 8`, `--embed MODEL`, `--no-embed`, `--local [MODEL] --ctx 16384 --port 8081 --think on|off|auto`. Env: `HARNESS_MODEL`, `LMSTUDIO_HOST`.

`--local` bật flash attention, KV cache q8_0, `--jinja` (bắt buộc để tool-calling hoạt động) và tắt khi thoát.
Log server: `/tmp/harness-llama-server.log`.

## Tool
Chỉ đọc (tự chạy): `read_file`, `list_dir`, `glob`, `grep`, `todo`.
Cần duyệt: `write_file`, `edit_file`, `move_file`, `delete_file`, `bash`, `web_fetch`.
Mọi đường dẫn bị giới hạn trong workspace. Tham số tool được kiểm tra và tự ép kiểu; lỗi trả về cho model để tự sửa.

## Bộ nhớ ngoài (RAG nhẹ)
Tìm lai: SQLite FTS5/BM25 (khớp từ, không phân biệt dấu) + embedding (khớp nghĩa), hợp nhất bằng RRF. Lưu ở `<workspace>/.harness/memory.db` (tự gitignore).
- **Lưu:** mỗi lượt hỏi/đáp và mỗi kết quả tool được cất vào bộ nhớ.
- **Đẩy ra khỏi context:** khi vượt ngân sách (`--ctx` × 2.2 ký tự), các lượt cũ bị bỏ khỏi context, chỉ giữ 2 lượt gần nhất.
- **Mớm đúng lúc:** mỗi lần bạn hỏi, hệ thống tự tìm 4 mục liên quan nhất và gắn tạm vào câu hỏi đó (không lưu vào lịch sử, nên context không phình).
- **Model chủ động:** tool `recall` (tra lại) và `remember` (ghi chú lâu dài). Bộ nhớ giữ qua các phiên.
- **Embedding:** tự bật `llama-server --embeddings` bằng CPU với model embedding trong `~/.lmstudio/models` (vd `bge-m3`), không tốn VRAM; ~25 ms/lần tìm. Ngưỡng cosine 0,5 (đo trên bge-m3). Mục chỉ khớp từ khoá phải chứa từ nguyên vẹn đúng dấu. Server embedding lỗi/tắt thì tự rơi về BM25; mục cũ chưa có vector được nhúng bù khi khởi động.
- Lệnh: `/memory` xem thống kê, `/forget` xoá hết; `--no-memory` để tắt.

## Tối ưu ngữ cảnh
- Kết quả tool cũ được lược khi ngữ cảnh vượt ngân sách (giữ 6 tin gần nhất).
- Chặn vòng lặp: gọi y hệt lần 3 thì cảnh báo, lần 4 thì từ chối.
- `HARNESS.md` / `AGENTS.md` / `CLAUDE.md` ở gốc workspace tự được nạp vào prompt hệ thống.

## Cấu trúc
`harness/client.py` client OpenAI-compatible · `tools.py` · `agent.py` vòng lặp · `ui.py` giao diện · `serve.py` chạy llama-server · `tests/`
(`python3 -m unittest discover -s tests`)

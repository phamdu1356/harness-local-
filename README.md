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
python3 -m harness --ask              # hỏi duyệt trước khi ghi/xoá/chạy lệnh (mặc định: TOÀN QUYỀN)
python3 -m harness --resume           # mở lại phiên gần nhất (hoặc --resume TÊN; lưu thêm bằng --session TÊN)
python3 -m harness eval --repeat 3    # chạy bộ eval, xem mục Eval bên dưới
```
Tuỳ chọn: `-m`, `--host`, `--temperature`, `--max-steps`, `--system "..."|@file`, `--text-tools`,
`--cpu --threads 8`, `--embed MODEL`, `--no-embed`, `--local [MODEL] --ctx 16384 --port 8081 --think on|off|auto`. Env: `HARNESS_MODEL`, `LMSTUDIO_HOST`.

`--local` bật flash attention, KV cache q8_0, `--jinja` (bắt buộc để tool-calling hoạt động) và tắt khi thoát.
Log server: `/tmp/harness-llama-server.log`.

## Tool
Mặc định agent **toàn quyền**, không hỏi duyệt; dùng `--ask` để hỏi trước các tool ghi/xoá/chạy lệnh/web/MCP.
`read_file`, `list_dir`, `glob`, `grep`, `todo`, `write_file`, `edit_file`, `move_file`, `delete_file`, `bash`, `web_fetch`, `web_search`, `delegate`.
Tool file bị giới hạn trong workspace, nhưng **`bash` thì không** (model có thể `cat` file ngoài workspace): đó là hệ quả của toàn quyền.
- `web_search`: DuckDuckGo (có thể bị chặn truy vấn tự động, khi đó tool báo lỗi rõ). Ổn định hơn: đặt `HARNESS_SEARX_URL` trỏ tới SearXNG của bạn.
- `delegate`: giao việc khảo sát chỉ-đọc cho subagent có context riêng, chỉ nhận lại kết luận.
- **MCP**: khai báo server trong `.harness/mcp.json` hoặc `~/.config/harness/mcp.json`
  (`{"mcpServers": {"tên": {"command": "...", "args": [...], "env": {...}}}}`); tool xuất hiện dạng `tên__tool`. Tham số tool được kiểm tra và tự ép kiểu; lỗi trả về cho model để tự sửa.

## Bộ nhớ ngoài (RAG nhẹ)
Tìm lai: SQLite FTS5/BM25 (khớp từ, không phân biệt dấu) + embedding (khớp nghĩa), hợp nhất bằng RRF. Lưu ở `<workspace>/.harness/memory.db` (tự gitignore).
- **Lưu:** mỗi lượt hỏi/đáp và mỗi kết quả tool được cất vào bộ nhớ.
- **Đẩy ra khỏi context:** khi vượt ngân sách (`--ctx` × 2.2 ký tự), các lượt cũ bị bỏ khỏi context, chỉ giữ 2 lượt gần nhất.
- **Mớm đúng lúc:** mỗi lần bạn hỏi, hệ thống tự tìm 4 mục liên quan nhất và gắn tạm vào câu hỏi đó (không lưu vào lịch sử, nên context không phình).
- **Model chủ động:** tool `recall` (tra lại) và `remember` (ghi chú lâu dài). Bộ nhớ giữ qua các phiên.
- **Embedding:** tự bật `llama-server --embeddings` bằng CPU với model embedding trong `~/.lmstudio/models` (vd `bge-m3`), không tốn VRAM; ~25 ms/lần tìm. Ngưỡng cosine 0,5 (đo trên bge-m3). Mục chỉ khớp từ khoá phải chứa từ nguyên vẹn đúng dấu. Server embedding lỗi/tắt thì tự rơi về BM25; mục cũ chưa có vector được nhúng bù khi khởi động.
- Lệnh: `/memory` xem thống kê, `/forget` xoá hết; `--no-memory` để tắt.


## Chạy việc dài một cách đáng tin (agent tự làm, harness kiểm chứng)
```bash
python3 -m harness -C DỰ_ÁN --resume build --session build \
  --verify "python3 -m unittest discover -s tests" --step-budget 30 --max-items 8 \
  --goal "<mục tiêu lớn kèm hiện trạng đã kiểm chứng>"
```
- **Nhốt trong một thư mục:** `bash` chạy qua Landlock (kernel Linux, không cần root): chỉ đọc/ghi được trong workspace (+ đọc `/usr`, `/lib`...). Tắt bằng `--no-sandbox`. Không dựng được sandbox thì `bash` tự tắt.
- **`--goal`:** agent lập kế hoạch (giai đoạn chỉ-đọc), harness NẮM kế hoạch và giao từng việc với ngân sách bước riêng; hết bước thì đánh dấu bị chặn và sang việc kế; hết bước mà suite đỏ thì bắt buộc sửa trước.
- **`--verify LỆNH`:** chạy khi agent báo xong; chỉ chặn lỗi MỚI so với mốc cố định; thất bại thì agent phải sửa tiếp (tối đa 3 vòng). Một việc chỉ tính xong khi file thật sự đổi và (nếu đổi `.py`) có test đi kèm; chưa đạt thì có 2 lần bổ sung rồi mới chặn (`--no-require-tests` để tắt).
- **Chống kẹt:** ngắt suy nghĩ bị lặp, nhắc khi không dùng `todo` / chạy script lẻ / 25 bước không sửa file, thử lại khi trả lời rỗng, báo file đã đổi kể từ lần đọc, nhật ký hành động tự ghi trong prompt (agent không "mất trí nhớ" khi context bị dọn).
- **Context:** đếm cả tham số `tool_calls`, dọn một lần xuống 70% ngân sách (giữ cache tiền tố của server), dọn được ngay trong một lượt dài.

## Context
- Đọc **số token thật** từ server (và context đang nạp của LM Studio/llama-server), học tỉ lệ ký tự/token để tính ngân sách (65% context dành cho hội thoại). `/context` xem chi tiết; chân mỗi lượt hiện `ctx 2.0k/63.5k`.
- Khi vượt ngân sách: lược kết quả tool cũ → đẩy cả lượt cũ ra, **tóm tắt bằng chính model** vào prompt hệ thống (`--no-summary` để tắt). Nội dung gốc vẫn nằm trong bộ nhớ ngoài.
- Server báo tràn context thì tự siết ngân sách và thử lại một lần.
- Phiên tự lưu vào `.harness/sessions/last.json` sau mỗi lượt.

## Eval
`python3 -m harness eval [--task TÊN] [--repeat N] [--label NHÃN] [--compare FILE.json] [--list]` (dùng chung cờ model/`--effort`/`--cpu`...).
9 tình huống: sửa lỗi, tìm định nghĩa, tạo file, sửa chính xác, đổi tên nhiều file, sửa cho test pass, nhớ sau khi bị đẩy khỏi context, không bịa file, đếm nơi gọi.
Chấm tự động (chạy code/so nội dung), lưu `.harness/evals/*.json`, `--compare` so với lần trước. Mã thoát 0 nếu đạt hết.

## Tối ưu ngữ cảnh (cũ)
- Kết quả tool cũ được lược khi ngữ cảnh vượt ngân sách (giữ 6 tin gần nhất).
- Chặn vòng lặp: gọi y hệt lần 3 thì cảnh báo, lần 4 thì từ chối.
- `HARNESS.md` / `AGENTS.md` / `CLAUDE.md` ở gốc workspace tự được nạp vào prompt hệ thống.

## Cấu trúc
`harness/client.py` client OpenAI-compatible · `tools.py` · `agent.py` vòng lặp · `ui.py` giao diện · `serve.py` chạy llama-server · `tests/`
(`python3 -m unittest discover -s tests`)

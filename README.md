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
Tuỳ chọn chính:
- Kết nối & model: `-m`/`--model`, `--host`, `--local [MODEL]`, `--cpu`, `--threads N`, `--ctx N`, `--port N`, `--effort low|medium|high`, `--think on|off|auto`, `--temperature F`, `--text-tools`. Env: `HARNESS_MODEL`, `LMSTUDIO_HOST`.
- Chế độ & ngữ cảnh: `-C`/`--cwd`, `-p`/`--prompt`, `--ask`, `--system "..."|@file`, `--max-steps N`, `--no-summary`, `--no-memory`, `--embed MODEL`, `--no-embed`, `--no-sandbox`.
- Kiểm chứng & chạy dài: `--verify LỆNH`, `--verify-rounds N`, `--goal`, `--step-budget N`, `--max-items N`, `--no-require-tests`, `--oracle SCRIPT`, `--stages`, `--start-stage N`, `--stage-rounds N`, `--resume [TÊN]`, `--session TÊN`.
- Hướng dẫn dự án: đặt `HARNESS.md`, `AGENTS.md` hoặc `CLAUDE.md` ở gốc workspace để tự nạp vào prompt hệ thống.

`--local` bật flash attention, KV cache `q8_0`, `--jinja` (cần cho tool-calling) và tự tắt server khi thoát.
Log server: `/tmp/harness-llama-server.log` (embedding: `/tmp/harness-embed-server.log`).

## Tool & Sandbox
Mặc định agent **toàn quyền**, không hỏi duyệt; dùng `--ask` để hỏi trước các tool ghi/xoá/chạy lệnh/web (`web_fetch`, `web_search`)/MCP. Subagent của `delegate` cũng phải qua duyệt khi bật `--ask`.

15 tool dựng sẵn:
- Đọc & tìm trong workspace: `read_file`, `list_dir`, `glob`, `grep`
- Ghi & sửa trong workspace: `write_file`, `edit_file`, `move_file`, `delete_file`
- Chạy lệnh & web: `bash`, `web_fetch`, `web_search`
- Lập kế hoạch, bộ nhớ & uỷ quyền: `todo`, `remember`, `recall`, `delegate`

Biên an toàn:
- **Tool file bị nhốt trong workspace và chặn `.harness/`:** mọi đường dẫn đều được `.resolve()` để chặn symlink hoặc đường dẫn trỏ ra ngoài workspace và chặn đọc/ghi `.harness/` (`mcp.json`, `memory.db`, `sessions/`).
- **`bash` chạy qua Linux Landlock** (`harness/sandbox.py`, tắt bằng `--no-sandbox`): chỉ đọc/ghi được trong workspace và đọc+thực thi các thư mục hệ thống tối thiểu (`/usr`, `/bin`, `/lib`, ...), chặn `/home`, `/etc/passwd`, `/tmp` hệ thống. Nếu kernel không hỗ trợ Landlock thì `bash` tự tắt (trừ khi bật `--no-sandbox`).
- **Giới hạn cần biết:** sandbox chỉ nhốt *file*, **không chặn mạng**, và **MCP server chạy ngoài sandbox** với quyền của user. Không cho agent toàn quyền đọc nội dung web không tin cậy khi workspace chứa dữ liệu nhạy cảm.
- **Biến môi trường:** `bash` tự lọc các biến có tên giống bí mật (`*TOKEN*`, `*SECRET*`, `*PASSWORD*`, `*API_KEY*`, `*_KEY`, `*AUTH*`, `*COOKIE*`...). Cần giữ biến nào thì đặt `HARNESS_ENV_ALLOW=TÊN1,TÊN2`.
- **`web_search`:** dùng DuckDuckGo (hoặc đặt `HARNESS_SEARX_URL` trỏ tới SearXNG).
- **`delegate`:** giao việc khảo sát chỉ-đọc cho subagent có context riêng, chỉ trả lại kết luận.
- **MCP:** khai báo trong `.harness/mcp.json` hoặc `~/.config/harness/mcp.json` (`{"mcpServers": {"tên": {"command": "...", "args": [...], "env": {...}}}}`); tool hiện dạng `tên__tool`, hỗ trợ kiểm tra/ép kiểu (kể cả schema nullable `["type", "null"]`).

## Bộ nhớ ngoài (RAG nhẹ)
Tìm lai: SQLite FTS5/BM25 (khớp từ, bỏ dấu tiếng Việt cả `đ`/`Đ`) + embedding (khớp nghĩa), hợp nhất bằng RRF. Lưu ở `<workspace>/.harness/memory.db` (tự gitignore).
- **Lưu:** mỗi lượt hỏi/đáp và mỗi kết quả tool được cất vào bộ nhớ.
- **Đẩy ra khỏi context:** khi vượt ngân sách context, các lượt cũ bị đẩy ra (giữ 1 lượt hiện tại trong context) và vẫn tìm lại được từ bộ nhớ ngoài.
- **Mớm đúng lúc:** mỗi lượt hỏi, hệ thống tự tìm 4 mục liên quan nhất từ bộ nhớ ngoài và gắn tạm vào câu hỏi gửi model (không lưu vào lịch sử).
- **Model chủ động:** tool `recall` (tra lại) và `remember` (ghi chú lâu dài), dùng chung qua các phiên.
- **Embedding:** tự bật `llama-server --embeddings` bằng CPU với model embedding trong `~/.lmstudio/models` (vd `bge-m3`), không tốn VRAM; ngưỡng cosine `0.5`. Mục chỉ khớp BM25 phải chứa từ khoá nguyên vẹn. Embedding lỗi/tắt thì tự rơi về BM25 và tự nhúng bù khi khởi động.
- Lệnh REPL: `/memory` xem thống kê, `/forget` xoá hết bộ nhớ ngoài; `--no-memory` để tắt.

## Chạy việc dài một cách đáng tin
```bash
python3 -m harness -C ~/code/du-an --resume build --session build \
  --verify "python3 -m unittest discover -s tests" --step-budget 30 --max-items 8 \
  --goal "<mục tiêu lớn kèm hiện trạng đã kiểm chứng>"
```
- **`--verify LỆNH` (dùng độc lập với `-p`/REPL hoặc kèm `--goal`):** tự chạy `LỆNH` khi agent báo xong; thất bại thì agent phải sửa tiếp tối đa `--verify-rounds` (mặc định `3`) vòng. Với `-p`, trả mã thoát `2` nếu hết bước hoặc kiểm tra cuối không đạt (`0` nếu đạt).
- **`--goal MỤC_TIÊU`:**
  1. Giai đoạn 1 lập kế hoạch chỉ-đọc bằng `todo`; harness nắm giữ danh sách việc chuẩn (model sửa `todo` cũng không lệch kế hoạch).
  2. Mỗi việc chạy riêng với `--step-budget`, bắt buộc phải có thay đổi file thật và (nếu đổi `.py` không phải test) phải có test `.py` đi kèm (tối đa 2 lần nhắc bổ sung, tắt bằng `--no-require-tests`).
  3. Nếu bật `--verify`, kiểm tra từng việc chỉ chặn lỗi *mới* so với mốc ban đầu, và việc chỉ tính `done` khi vượt qua kiểm tra; cuối `--goal`, chạy **KIỂM TRA CUỐI CÙNG** không nới theo mốc (mã thoát `0` chỉ khi mọi việc đều `done` và kiểm tra cuối đạt, ngược lại `2`). Hỗ trợ `--goal --resume` để tiếp tục kế hoạch còn dở.
- **Bền khi server lỗi:** lỗi mạng/luồng hỏng được thử lại một lần; lỗi giữa chừng vẫn lưu phiên (`--goal` lưu sau mỗi việc, `--stages` sau mỗi giai đoạn) để `--resume`. Ở REPL, lỗi server báo lỗi và thu hồi lượt treo; `Ctrl+C` khi đang chạy cũng dọn lịch sử dở dang và lưu lại.
- **Chống kẹt:** ngắt suy nghĩ lặp/quá dài, chặn gọi lặp y hệt (cảnh báo lần 3, từ chối lần 4), nhắc khi không dùng `todo` / chạy `python3 -c` / 25 bước không sửa file, thử lại khi trả lời rỗng, báo file đã đổi kể từ lần đọc, và giữ nhật ký hành động trong prompt hệ thống.

## Nghiệm thu độc lập theo giai đoạn (`--oracle` + `--stages`)
Giao dự án lớn cho model nhỏ đáng tin hơn khi có **script nghiệm thu đặt ngoài vùng agent truy cập** (agent không đọc/sửa được):
```bash
python3 -m harness -C /tmp/du-an --session v2 --stages --oracle examples/oracle/oracle.py \
  --step-budget 40 --stage-rounds 4 --verify 'python3 -m unittest discover -s tests; rc=$?; [ $rc -eq 5 ] && exit 0; exit $rc'
```
- Đường dẫn `--oracle` tương đối được phân giải từ thư mục chạy lệnh trước khi chuyển vào `-C`. Oracle ví dụ (`examples/oracle/oracle.py`) chạy app thật trong sandbox với LLM/Telegram giả theo giao ước `--list` và `--root DIR --upto K`.
- Harness chạy lần lượt từng giai đoạn, nghiệm thu bằng oracle (gồm hồi quy các giai đoạn trước); chưa đạt giai đoạn trước thì không sang giai đoạn sau.
- Mã thoát `--stages`: `0` chỉ khi đạt tới giai đoạn cuối cùng từ `--list` và mọi giai đoạn đã chạy đều đạt; ngược lại `2`.

## Context & Phiên
- Đọc `prompt_tokens` thực từ server (và context nạp từ LM Studio `/api/v0/models` hoặc llama-server `/props`, mặc định `--local` là `16384`), tự hiệu chỉnh tỉ lệ ký tự/token (khởi đầu `3.0` ký tự/token) cho ngân sách **50%** context dành cho hội thoại. `/context` xem chi tiết; chân mỗi lượt hiện `ctx ...`.
- Khi vượt ngân sách: lược kết quả/tham số tool cũ (giữ 6 tin không phải system gần nhất) → đẩy các lượt cũ ra khỏi context xuống 70% ngân sách (luôn giữ **1** lượt hiện tại, giữ tiền tố để tận dụng KV cache) → **tóm tắt bằng chính model** vào prompt hệ thống (`--no-summary` để tắt).
- Server báo tràn context thì tự siết ngân sách và thử lại một lần.
- Phiên tự lưu vào `<workspace>/.harness/sessions/last.json` sau mỗi lượt (và `/clear` cũng cập nhật bản lưu).

## Eval
`python3 -m harness eval [--task TÊN] [--repeat N] [--label NHÃN] [--compare FILE.json] [--out DIR] [--list]` (nhận các cờ chung như `-m`, `--host`, `--local`, `--effort`, `--cpu`...).
10 tình huống: `fix_bug`, `find_def`, `create_file`, `precise_edit`, `rename_multi`, `fix_until_tests_pass`, `memory_recall`, `missing_file`, `sandbox_escape`, `count_callers`.
Chấm tự động (chạy code/so nội dung), lưu `.harness/evals/*.json`, `--compare` so với lần trước. Mã thoát `0` nếu đạt hết.

## Cấu trúc
- `harness/__main__.py`, `cli.py`: điểm vào, phân tích cờ, REPL và điều phối chế độ
- `harness/client.py`, `serve.py`, `embed.py`: client OpenAI-compatible, tự chạy `llama-server`, embedding client
- `harness/tools.py`, `mcp.py`, `sandbox.py`, `_sandbox_exec.py`: tool workspace/web, MCP client, sandbox Landlock & lọc env
- `harness/agent.py`, `memory.py`: vòng lặp agent, nén context/tóm tắt/nhật ký, bộ nhớ ngoài FTS5 + vector
- `harness/goal.py`, `stages.py`, `evals.py`, `ui.py`: chế độ `--goal`, nghiệm thu `--stages`, bộ eval, giao diện terminal
- `tests/`: kiểm thử (`python3 -m unittest discover -s tests`)

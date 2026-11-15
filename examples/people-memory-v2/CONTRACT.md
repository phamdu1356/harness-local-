# CONTRACT: giao ước mà hệ thống nghiệm thu kiểm tra

Hệ thống nghiệm thu (oracle) chạy ứng dụng của bạn THẬT, gửi yêu cầu HTTP THẬT, dùng một LLM giả và một Telegram giả chạy cục bộ,
rồi báo chính xác chỗ nào sai. Bạn tự quyết định kiến trúc bên trong; chỉ cần tuân thủ giao ước dưới đây.

## 1. Chạy và cấu hình
- Chạy từ thư mục dự án: `python3 -m people_memory` (cần `people_memory/__init__.py` và `people_memory/__main__.py`).
- Cấu hình hoàn toàn bằng biến môi trường (có giá trị mặc định):

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `PM_HOST` | `127.0.0.1` | Địa chỉ nghe. Mặc định phải là 127.0.0.1, không phải mọi giao diện |
| `PM_PORT` | `8765` | Cổng web |
| `PM_DATA_DIR` | `./data` | Thư mục dữ liệu: `people.db` (SQLite) và `backups/`. Tự tạo nếu chưa có |
| `PM_LLM_URL` | `http://127.0.0.1:1234/v1` | Gốc API kiểu OpenAI (gọi `POST {PM_LLM_URL}/chat/completions`, `stream` = false) |
| `PM_LLM_MODEL` | `local` | Tên model gửi trong body |
| `PM_LLM_TIMEOUT` | `20` | Giây chờ LLM; quá hạn hoặc không kết nối được coi là "LLM không sẵn sàng" |
| `PM_SCHEDULER_INTERVAL` | `60` | Giây giữa các lần luồng nền xả hàng đợi, đẩy nhắc việc, kiểm tra sao lưu |
| `PM_BACKUP_INTERVAL` | `86400` | Giây tối thiểu giữa hai bản sao lưu tự động (khi khởi động nếu chưa có bản nào trong khoảng này thì sao lưu ngay) |
| `TELEGRAM_BOT_TOKEN` | (trống) | Trống = tắt Telegram |
| `TELEGRAM_CHAT_ID` | (trống) | Chat id DUY NHẤT được phép |
| `TELEGRAM_API_BASE` | `https://api.telegram.org` | Gốc API Telegram (test đổi sang máy chủ giả) |
| `PM_TELEGRAM_POLL` | `1.0` | Giây nghỉ giữa các lần hỏi `getUpdates` |

- Mọi tiến trình khởi động lại phải đọc lại dữ liệu từ `PM_DATA_DIR` (không mất gì).
- KHÔNG được gọi mạng tới nơi nào khác ngoài `PM_LLM_URL` và `TELEGRAM_API_BASE`.
- Mọi phản hồi JSON dùng UTF-8, header `Content-Type: application/json; charset=utf-8` (trừ trang chủ).

## 2. Giao diện web
- `GET /` trả HTML (`Content-Type: text/html`) có ô nhập (`<textarea` hoặc `<input`) và `<form` hoặc nút gửi, tiếng Việt. Người dùng gõ một câu rồi gửi tới `/api/message`.
- `GET /api/health` -> `{"ok": true}`.

## 3. Nhập một câu tự nhiên (tiêu chí 1)
`POST /api/message` body `{"text": "...", "at": "2026-11-15T09:00:00" (tuỳ chọn, mặc định bây giờ)}`

- `text` rỗng/thiếu -> HTTP 400 `{"error": "..."}`.
- Gọi LLM: gửi `text` gốc, yêu cầu LLM trả về DUY NHẤT một JSON theo lược đồ trích xuất bên dưới; đọc `choices[0].message.content`
  (có thể bọc trong ```json ... ```, hãy chịu được). JSON không hợp lệ coi như lỗi (xem mục 8).
- Thành công -> HTTP 200 `{"status": "saved", "people": [{"id": 1, "name": "My", "created": true}], "facts_added": 2}`.
- LLM không sẵn sàng (không kết nối được, quá hạn, HTTP lỗi, JSON hỏng) -> HTTP 202 `{"status": "queued", "queue_id": 7}`; tin được lưu vào hàng đợi (xem mục 8). KHÔNG được mất tin.
- Phải xong dưới 3 giây khi LLM phản hồi nhanh.

**Lược đồ trích xuất** (LLM trả về, ứng dụng phân tích):
```json
{"people": [
  {"name": "My", "aliases": ["Mỹ", "em My"], "relation": "người yêu",
   "facts": [
     {"kind": "preference", "key": "food:onion", "text": "không thích ăn hành", "date": null, "recurring": null},
     {"kind": "allergy",    "key": "allergy:seafood", "text": "dị ứng hải sản", "date": null, "recurring": null},
     {"kind": "event",      "key": "birthday", "text": "sinh nhật", "date": "2026-12-20", "recurring": "yearly"}
   ]}
]}
```
`kind` thuộc: `preference`, `allergy`, `health`, `event`, `place`, `gift_idea`, `info`. `key` là chủ đề ổn định (dùng để phát hiện cập nhật). `date` dạng `YYYY-MM-DD` hoặc null. `aliases`, `relation` có thể thiếu/null.

**Quy tắc lưu**
- Chuẩn hoá tên/biệt danh để so khớp: viết thường, bỏ dấu tiếng Việt (đ->d), gộp khoảng trắng. Người trong kết quả được coi là CÙNG MỘT NGƯỜI với người đã có nếu tên chuẩn hoá của họ trùng tên chuẩn hoá hoặc một biệt danh chuẩn hoá của người đó (và ngược lại: biệt danh mới trùng tên/biệt danh cũ). Khi trùng thì GỘP biệt danh mới, không tạo người mới. Không trùng thì tạo người mới.
- Mỗi tin nhắn nhắc tới một người cập nhật `last_interaction_at` của người đó = `at` (hoặc bây giờ), nếu mới hơn giá trị cũ.
- Fact mới cùng (người, `key`):
  - cùng `text` với fact đang `active` -> không thêm lần nữa;
  - khác `text` -> fact cũ chuyển `status` = `superseded` (và ghi `superseded_at`), GIỮ NGUYÊN trong dữ liệu; fact mới `active`.
- Fact không có `key` thì dùng chuẩn hoá của `text` làm key.

## 4. Đọc hồ sơ (tiêu chí 2)
- `GET /api/people` -> `{"people": [{"id": 1, "name": "My", "aliases": ["Mỹ"], "relation": "người yêu", "last_interaction_at": "2026-11-15T09:00:00"}]}`
- `GET /api/people/<id>` -> `{"id", "name", "aliases", "relation", "last_interaction_at", "facts": [{"id", "kind", "key", "text", "date", "recurring", "status": "active" | "superseded", "created_at", "superseded_at": null | "..."}]}`; id không có -> 404.
- `POST /api/ask` body `{"question": "My dị ứng gì?"}` -> `{"answer": "...", "people": [1], "facts": [ ...các fact ACTIVE của người được nhắc... ]}`
  - Tìm người được nhắc bằng cách quét câu hỏi theo tên/biệt danh chuẩn hoá (không cần LLM).
  - Không có người nào được nhắc -> `{"answer": "Mình chưa có thông tin về ai được nhắc trong câu hỏi.", "people": [], "facts": []}` và KHÔNG gọi LLM, KHÔNG bịa.
  - Có người: đưa các fact active vào prompt cho LLM để diễn đạt câu trả lời (chỉ dựa trên fact đó). LLM không sẵn sàng -> vẫn trả HTTP 200 với `answer` ghép từ fact và `facts` đầy đủ.

## 5. Chủ động đúng lúc (tiêu chí 3)
`GET /api/reminders?now=2026-11-15T09:00:00` (`now` mặc định bây giờ) -> `{"reminders": [{"type", "person_id", "due", "days", "message"}]}`. Phải ổn định: gọi hai lần với cùng `now` cho kết quả như nhau.

- `birthday`: fact kind `event` có key `birthday`, lặp hằng năm. Nhắc khi còn 0..3 ngày tới lần sinh nhật kế tiếp (tính cả năm sau). `days` = số ngày còn lại; `due` = ngày sinh nhật kế tiếp (`YYYY-MM-DD`). `message` tiếng Việt, phải chứa tên người và đúng số ngày (ví dụ còn 3 ngày -> chứa "3 ngày"; còn 0 -> chứa "hôm nay"; tuyệt đối không viết "tối nay" khi còn nhiều ngày).
- `event`: fact kind `event` khác sinh nhật, có `date` từ hôm nay đến ngày mai (0..1 ngày). `days` là số ngày còn lại, `message` chứa tên người và tên sự kiện (`text`).
- `no_contact`: người có `last_interaction_at` đã quá ngưỡng. Ngưỡng theo `relation` chuẩn hoá: chứa "me", "bo", "ba", "ma", "gia dinh", "bo me" (từ nguyên vẹn) -> 21 ngày; "nguoi yeu" -> 7 ngày; còn lại -> 60 ngày. `days` = số ngày nguyên đã trôi qua kể từ `last_interaction_at`; `message` chứa tên và số đó. Không nhắc người vừa liên lạc trong ngưỡng. Người chưa từng có `last_interaction_at` thì không nhắc.
- `weekly_plan`: khi `now` là thứ Sáu (weekday 4) từ 17:00 trở đi: đúng 1 nhắc loại này với `person_id` null và `message` gợi ý rủ ai đi chơi cuối tuần.
- `POST /api/reminders/push?now=...` đẩy các nhắc tới Telegram (xem mục 9). Mỗi nhắc (loại, người, `due`) chỉ đẩy MỘT lần: lần gọi thứ hai không gửi lại. Trả `{"sent": n}`. Telegram tắt -> `{"sent": 0}`.

## 6. Soạn tin và gợi ý đi chơi (tiêu chí 3)
- `POST /api/compose` body `{"person": "My", "situation": "Đang bực lắm, muốn nhắn là tại sao lúc nào cũng trễ hẹn"}` -> `{"options": {"gentle": "...", "firm": "...", "humor": "..."}}`.
  - Prompt gửi LLM phải chứa quan hệ và các fact ACTIVE của người đó cùng `situation`, và yêu cầu LLM trả JSON có đúng ba khoá `gentle`, `firm`, `humor`.
  - Người không tồn tại -> 404. LLM không sẵn sàng -> 503 `{"error": "..."}`.
- `POST /api/plan` body `{"person": "My", "request": "Tối thứ 7 đi đâu quanh Hà Đông, trời lạnh, dưới 500k", "weather": "lạnh" (tuỳ chọn)}` -> `{"stops": [{"title": "...", "reason": "..."}, {"title": "...", "reason": "..."}], "excluded": ["..."]}`.
  - Prompt phải chứa các sở thích/dị ứng ACTIVE của người đó, yêu cầu `request`, `weather` nếu có, và danh sách các địa điểm đã đi (fact kind `place`, lấy `text`) kèm yêu cầu rõ ràng KHÔNG gợi ý lại; yêu cầu LLM trả JSON có khoá `stops` (đúng 2 phần tử, mỗi phần tử `title`, `reason`).
  - Sau khi nhận kết quả, ứng dụng loại bỏ stop có `title` trùng (không phân biệt hoa/thường, bỏ dấu) một địa điểm đã đi; `excluded` là danh sách địa điểm đã đi. Kết quả cuối vẫn phải có đúng 2 stop (hãy yêu cầu LLM lần nữa nếu cần, tối đa 2 lần thử thêm); không đủ thì HTTP 502.
  - Người không tồn tại -> 404. LLM không sẵn sàng -> 503.

## 7. (không dùng)

## 8. Chịu lỗi (tiêu chí 4)
- Hàng đợi bền (SQLite): tin chưa xử lý được nằm trong hàng đợi và còn nguyên sau khi khởi động lại.
- `GET /api/queue` -> `{"pending": 1, "items": [{"id": 7, "text": "...", "at": "...", "attempts": 2}]}`.
- `POST /api/queue/drain` xử lý lần lượt các tin theo thứ tự cũ -> mới, mỗi tin qua đúng quy trình mục 3 (dùng `at` đã lưu). Tin xử lý thành công thì bỏ khỏi hàng đợi; thất bại thì GIỮ LẠI, tăng `attempts`, và dừng ngay lượt xả (LLM còn tắt). Trả `{"processed": n, "remaining": m}`.
- Luồng nền cũng tự xả hàng đợi mỗi `PM_SCHEDULER_INTERVAL` giây.
- Sao lưu: `POST /api/backup` -> `{"path": "<PM_DATA_DIR>/backups/people-YYYYmmdd-HHMMSS.db"}` là một file SQLite hợp lệ chứa đủ dữ liệu hiện tại (dùng `sqlite3` backup API hoặc tương đương, an toàn khi đang ghi). `GET /api/backups` -> `{"backups": [{"path": "...", "size": 12345}]}`. Tự động sao lưu theo `PM_BACKUP_INTERVAL` (và khi khởi động nếu chưa có bản nào trong khoảng đó). Giữ tối đa 14 bản gần nhất.
- Khôi phục: chép một file backup đè lên `PM_DATA_DIR/people.db` khi ứng dụng đang tắt (ghi trong README).

## 9. Telegram, khoá một tài khoản (tiêu chí 1 và 5)
Khi `TELEGRAM_BOT_TOKEN` được đặt, một luồng nền:
- Hỏi `POST {TELEGRAM_API_BASE}/bot{token}/getUpdates` (JSON body `{"offset": n, "timeout": 0}`), phản hồi `{"ok": true, "result": [ {"update_id": 1, "message": {"message_id": 1, "chat": {"id": 1001}, "from": {"id": 1001}, "text": "..."}} ]}`; dùng `offset` = `update_id` lớn nhất đã thấy + 1 để không xử lý lại.
- Chỉ xử lý update có `message.chat.id` bằng `TELEGRAM_CHAT_ID` (so sánh dạng chuỗi) và có `text`: chạy đúng quy trình mục 3 rồi trả lời bằng `POST {TELEGRAM_API_BASE}/bot{token}/sendMessage` body `{"chat_id": ..., "text": "..."}` (xác nhận ngắn tiếng Việt; nếu tin vào hàng đợi thì nói đã xếp hàng chờ).
- Update của chat id khác: BỎ QUA HOÀN TOÀN: không lưu, không gọi LLM, không trả lời.
- Telegram lỗi mạng thì thử lại sau, không làm sập ứng dụng. Nếu `TELEGRAM_CHAT_ID` trống thì không xử lý update nào (tránh mở cửa cho mọi người).

## 10. Riêng tư (tiêu chí 5)
- Nghe trên `PM_HOST` (mặc định `127.0.0.1`).
- Không có URL nào khác ngoài `PM_LLM_URL`/`TELEGRAM_API_BASE` (và mặc định của chúng) trong mã nguồn.
- Dữ liệu chỉ ở `PM_DATA_DIR`. Token không được ghi ra log hay vào DB.

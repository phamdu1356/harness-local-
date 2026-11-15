# Quy ước dự án people-memory (bản làm lại theo giao ước)
- Python 3.10+, CHỈ thư viện chuẩn (sqlite3, http.server, json, urllib, threading, unittest). Không pip.
- Gói `people_memory/` phải chạy được bằng `python3 -m people_memory` (cần `__init__.py` và `__main__.py`) từ thư mục này.
- Đọc CONTRACT.md: đó là giao ước HTTP/cấu hình mà hệ thống nghiệm thu độc lập kiểm tra. Hệ thống nghiệm thu chạy ứng dụng của bạn
  thật, gửi yêu cầu thật và báo lỗi cụ thể: sửa theo thông báo lỗi đó, đừng đoán.
- Mọi dữ liệu trong SQLite cục bộ. Không gửi gì ra ngoài, chỉ gọi PM_LLM_URL và TELEGRAM_API_BASE (xem CONTRACT.md).
- Mọi test của bạn trong tests/ dùng thư mục tạm riêng và tự dọn, không gọi LLM thật, không để file rác. Mỗi tính năng có test.
- Giữ cấu trúc đơn giản: ít file, mỗi file một việc, không có bản sao/nháp trùng chức năng.
- Việc hiểu một câu tự nhiên (ai, sở thích, dị ứng, sự kiện, ngày...) PHẢI do LLM làm qua HTTP tới PM_LLM_URL theo CONTRACT.md mục 3.
  Không được tự trích xuất bằng regex hay từ khoá: hệ thống nghiệm thu dùng một LLM giả và sẽ đếm số yêu cầu mà ứng dụng gửi tới nó.

# Tiêu chí nghiệm thu (nguyên văn từ người dùng)

Một app Personal Relationship & Social Memory CRM (hay bất kỳ Agent cá nhân nào chạy local) được coi là đạt yêu cầu khi đáp ứng đủ 5 tiêu chí cốt lõi sau:

1. Ma sát nhập liệu bằng 0 (Zero-Friction Input)

* Không bắt người dùng phải mở app, điền form hay chọn danh mục thủ công.

* Nhắn một câu tự nhiên (hoặc gửi voice/ảnh) qua Telegram là hệ thống tự hiểu, tự phân loại và lưu đúng người, đúng việc trong dưới 3 giây.

2. Dữ liệu sạch và nhất quán (Data Integrity)

* Không trùng lặp: Tự nhận diện được biệt danh và tên thật là cùng một người.

* Không mâu thuẫn: Biết tự cập nhật trạng thái mới (ví dụ: từ "thích ăn cay" sang "đang đau dạ dày phải kiêng cay") mà không làm mất lịch sử cũ.

* Truy xuất chính xác: Hỏi đến ai là lôi ra đúng sự thật về người đó, tuyệt đối không bịa đặt (hallucination).

3. Chủ động đúng lúc, không làm phiền (Proactive & Contextual)

* Không chỉ ngồi chờ hỏi mới trả lời, mà biết tự động nhắc việc trước giờ hẹn, trước ngày sinh nhật/kỷ niệm hoặc khi một mối quan hệ quá lâu chưa tương tác.

* Gợi ý đi chơi, tặng quà hay soạn tin nhắn phải khớp với hồ sơ tính cách/sở thích của đối phương và loại trừ những gì đã làm rồi.

4. Chịu lỗi và hoạt động bền bỉ (Resilience)

* Khi bạn tắt LM Studio (để chơi game hoặc tắt máy), hệ thống không bị mất dữ liệu: tin nhắn gửi qua điện thoại vẫn được xếp vào hàng đợi (Queue) và tự động xử lý bù ngay khi server AI bật lại.

* Tự động sao lưu (backup) database định kỳ, không bao giờ để mất ký ức đã lưu.

5. Bảo mật tuyệt đối (Strict Privacy)

* Khóa cứng quyền truy cập chỉ cho duy nhất tài khoản của bạn.

* Toàn bộ dữ liệu nhạy cảm (tâm sự, thói quen, lịch trình cá nhân) chỉ nằm trong ổ cứng máy bạn và chạy qua model local, không rò rỉ ra bất kỳ dịch vụ bên thứ ba nào.

Tóm lại: Nhập liệu lười nhất có thể – Nhớ chính xác tuyệt đối – Nhắc đúng thời điểm – Chạy ổn định kể cả khi tắt AI – Dữ liệu nằm trọn trong tay bạn.

# Yêu cầu (nguyên văn từ người dùng)

1. Những vấn đề đời thường nó giải quyết ngay lập tức

1. Ghi nhớ mọi chi tiết nhỏ về người thân, người yêu, bạn bè (Zero-Effort People Memory):
   * Bình thường đi chơi, đi ăn hay nói chuyện, người kia hay buột miệng nói những câu như: "Em không thích ăn hành", "Dạo này đang đau lưng", "Thích cái tai nghe màu trắng kia nhưng đắt quá", hoặc "Tuần sau thi môn X".
   * Bạn chỉ cần nhắn nhanh 1 câu lộn xộn vào Telegram cho Agent (hoặc nói bằng giọng nói): "My bảo thích cuốn sách Y và dị ứng hải sản" hoặc "Thằng Hùng mới chuyển trọ sang Cầu Giấy".
   * Agent tự phân loại theo từng người, lưu vào hồ sơ riêng. Đến sát ngày sinh nhật, kỷ niệm, hoặc lúc bạn sắp đi ăn/mua quà với họ, Agent tự động nhắn nhắc:
      * "Tối nay đi ăn với My: nhớ dặn quán không bỏ hành, tránh đồ hải sản. Ngoài ra tháng trước My có nhắc thích cuốn sách Y — làm quà dịp này là chuẩn bài."
   * Đến ngày người kia thi cử hoặc có sự kiện quan trọng, Agent nhắc bạn nhắn 1 câu hỏi thăm đúng lúc.
2. "Phanh cảm xúc" & Soạn tin nhắn khéo léo (Tone & Conflict De-escalator):
   * Lúc đang cãi nhau, hiểu lầm, hoặc cần từ chối một lời nhờ vả khó xử (mượn tiền, nhờ làm hộ việc, từ chối đi nhậu), bạn rất dễ nhắn những câu cộc lốc gây mất lòng hoặc cả nể đồng ý rồi về tự bực mình.
   * Bạn chỉ cần gõ thô suy nghĩ thật của mình cho Agent: "Đang bực lắm, muốn nhắn là tại sao lúc nào cũng trễ hẹn như thế" hoặc "Từ chối khéo thằng bạn mượn 2 triệu vì tháng này kẹt".
   * Agent phân tích ngữ cảnh mối quan hệ và đưa ra 3 phương án trả lời theo 3 sắc thái:
      * Nhẹ nhàng, giữ hòa khí (dùng cho người yêu, người thân).
      * Thẳng thắn, lịch sự nhưng kiên quyết (dùng cho bạn bè, đồng nghiệp).
      * Hài hước để hạ nhiệt căng thẳng.
3. Lên kịch bản đi chơi / hẹn hò / tụ tập "đo ni đóng giày" (Context-Aware Hangout Planner):
   * Cuối tuần muốn rủ người yêu hoặc nhóm bạn đi chơi nhưng lại rơi vào vòng lặp "Đi đâu cũng được / Ăn gì cũng được".
   * Bạn chỉ cần nhắn: "Tối thứ 7 rủ My đi đâu quanh Hà Đông/Thanh Xuân, trời hơi lạnh, ngân sách dưới 500k, tránh mấy quán ồn ào".
   * Agent đối chiếu với lịch sử những chỗ đã đi (để không bị trùng quán cũ), sở thích ăn uống đã lưu của người đó và thời tiết thực tế, sau đó vạch sẵn một lịch trình 2 chặng (Ăn tối ở đâu + Cafe/dạo bộ ở đâu) kèm lý do chọn hợp lý.
4. Nhắc nhở quan tâm định kỳ (Relationship Ping):
   * Bận rộn học hành, làm việc khiến bạn vô tình 2–3 tháng không gọi điện về cho bố mẹ hoặc mất liên lạc với những người bạn tốt.
   * Agent tự theo dõi tần suất tương tác (dựa trên ghi chú của bạn) và nhẹ nhàng nhắc: "Đã 3 tuần rồi bạn chưa gọi về cho mẹ" hoặc "Lâu rồi chưa hỏi thăm thằng Nam, cuối tuần rủ nó làm cốc cafe đi."

2. Chiều sâu kỹ thuật (Đời thường nhưng kiến trúc rất "chất")

* 100% Riêng tư tuyệt đối: Toàn bộ bí mật, sở thích, chuyện cãi vã hay thông tin cá nhân của người thân đều nằm trong SQLite + Vector DB cục bộ trên máy của bạn, xử lý hoàn toàn qua LM Studio local — không bao giờ bị đẩy lên cloud của bên thứ ba.
* Graph / Entity-Relationship Memory: Thay vì lưu văn bản phẳng, bạn xây dựng cấu trúc đồ thị thực thể đơn giản trong SQLite (`People` <-> `Preferences`, `Events`, `Interactions`, `Gift_Ideas`). Khi bạn nhắc đến một cái tên, Agent tự động truy xuất toàn bộ lịch sử liên quan đến người đó để làm ngữ cảnh (Context Injection) trước khi trả lời.
* Proactive Temporal Trigger (Bộ kích hoạt theo thời gian & ngữ cảnh): Kết hợp bộ lập lịch thông minh để Agent không chỉ ngồi chờ bạn hỏi, mà biết chủ động nhắn tin trước vào đúng thời điểm vàng (trước giờ hẹn 2 tiếng, trước ngày kỷ niệm 3 ngày, hoặc vào tối thứ 6 hàng tuần).

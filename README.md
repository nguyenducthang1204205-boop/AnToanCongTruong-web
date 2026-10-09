# ⛑️ Hệ thống giám sát an toàn công trường (bản web)

Đồ án môn Học máy (Nhóm 1). Mô hình **YOLOv11n** được tinh chỉnh trên bộ dữ liệu PPE để kiểm tra **từng công nhân** có đủ 4 trang bị bắt buộc hay không: **mũ bảo hộ, găng tay, áo bảo hộ, giày bảo hộ**. Ứng dụng cũng cảnh báo khi phát hiện công nhân để tay trần.

Bản web này chạy trên [Streamlit Community Cloud](https://streamlit.io/cloud), chỉ dùng CPU.

## Chức năng

| Tab | Mô tả |
|---|---|
| 🖼️ Ảnh | Tải một hoặc nhiều ảnh (hoặc chọn ảnh mẫu). Kết quả gồm ảnh đã đánh dấu, trạng thái AN TOÀN / CẢNH BÁO, bảng ✔/✘ cho từng công nhân, nút tải ảnh kết quả và nhật ký CSV |
| 🎞️ Video | Xem video gốc cạnh video đã phân tích (30 giây đầu). Kết quả gồm video đã đánh dấu, ảnh các khoảnh khắc vi phạm và nhật ký cảnh báo |
| 📷 Camera | **Video trực tiếp** qua WebRTC (mặc định). Nếu mạng chặn WebRTC thì dùng **Tự chụp mỗi giây**. Bảng trạng thái, ảnh cảnh báo gần nhất và nhật ký tự cập nhật |

### Khi chế độ Video trực tiếp không lên hình

Trên Streamlit Community Cloud, WebRTC chỉ dùng STUN thường **không truyền được hình**, nên cần thêm máy chủ chuyển tiếp **TURN**. App sẽ tự hiện cảnh báo nếu sau khoảng 8 giây máy chủ chưa nhận được hình nào. Cách thêm TURN (chọn một), vào **Manage app → Settings → Secrets** rồi bấm **Save**:

- **Hugging Face (miễn phí 10 GB/tháng, dễ nhất):** tạo token loại *Read* ở https://huggingface.co/settings/tokens, rồi thêm dòng:
  ```toml
  HF_TOKEN = "hf_xxxxxxxxxxxxxxxx"
  ```
- **Cloudflare Realtime TURN (miễn phí 1000 GB/tháng):** trong Cloudflare Dashboard → *Realtime* → *TURN Server* → tạo khóa, rồi thêm:
  ```toml
  CLOUDFLARE_TURN_KEY_ID = "..."
  CLOUDFLARE_TURN_KEY_API_TOKEN = "..."
  ```

Nếu chưa thêm được TURN, bấm **"Không lên hình? Chuyển sang chế độ tự chụp mỗi giây"**. Chế độ này chạy qua kết nối web thường.

Cũng có thể tự khai báo danh sách STUN/TURN bất kỳ (ví dụ [Metered](https://www.metered.ca/stun-turn)):

```toml
[[ice_servers]]
urls = ["stun:stun.l.google.com:19302"]

[[ice_servers]]
urls = ["turn:<máy-chủ>:80", "turn:<máy-chủ>:443?transport=tcp"]
username = "<tên đăng nhập>"
credential = "<mật khẩu>"
```

Muốn hình mượt nhất (khoảng 28 khung hình/giây) khi demo, hãy dùng app desktop `giao_dien.py` trong repo đồ án chính, chạy trên laptop có GPU.

## Cách hoạt động

1. YOLOv11n phát hiện 9 lớp: `Gloves, Helmet, Person, Safety Boot, Safety Vest, bare-arms, no-boot, no-helmet, no-vest`.
2. Mỗi trang bị được gán cho công nhân (`Person`) có khung chứa tâm của nó.
3. Một công nhân **đạt chuẩn** khi có đủ 4 trang bị và không có lớp "thiếu" nào. Bộ dữ liệu không có lớp `no-gloves`, nên thiếu găng tay được suy ra từ việc không phát hiện găng tay.

Logic kiểm tra nằm trong `notebook_code.py` (bản sao phần code của notebook đồ án). Trọng số mô hình nằm ở `runs/detect/train/weights/best.pt`.

## Chạy trên máy

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Nguồn

- Mô hình: [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) (AGPL-3.0)
- Dữ liệu: bộ PPE xuất từ Roboflow (CC BY 4.0)

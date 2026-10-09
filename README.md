# ⛑️ Hệ thống giám sát an toàn công trường (bản web)

Đồ án môn Học máy (Nhóm 1). Mô hình **YOLOv11n** được tinh chỉnh trên bộ dữ liệu PPE để kiểm tra **từng công nhân** có đủ 4 trang bị bắt buộc hay không: **mũ bảo hộ, găng tay, áo bảo hộ, giày bảo hộ**. Ứng dụng cũng cảnh báo khi phát hiện công nhân để tay trần.

Bản web này chạy trên [Streamlit Community Cloud](https://streamlit.io/cloud), chỉ dùng CPU.

## Chức năng

Ứng dụng có hai trang (thanh điều hướng ở trên cùng): **Giám sát** gồm ba tab dưới đây, và **Hiệu năng mô hình**.

| Tab | Mô tả |
|---|---|
| 🖼️ Ảnh | Tải một hoặc nhiều ảnh (hoặc chọn ảnh mẫu). Nhiều ảnh thì có phần tổng hợp: tỷ lệ tuân thủ và biểu đồ trang bị bị thiếu. Kết quả từng ảnh gồm ảnh đã đánh dấu, trạng thái AN TOÀN / CẢNH BÁO, bảng ✔/✘ cho từng công nhân, nút tải ảnh kết quả và nhật ký CSV |
| 🎞️ Video | Phân tích toàn bộ video, nên video kết quả dài đúng bằng video gốc. Hai video đặt cạnh nhau và phát cùng lúc (phát, tạm dừng, tua ở một bên thì bên kia làm theo). Kết quả gồm video đã đánh dấu, biểu đồ diễn biến vi phạm theo từng giây, biểu đồ trang bị bị thiếu, phần **Mô hình trên video này** (độ tin cậy theo thời gian và theo lớp, thời gian suy luận, tỷ lệ tuân thủ, dòng thời gian vi phạm theo trang bị, bản đồ nhiệt vị trí vi phạm), ảnh các khoảnh khắc vi phạm và nhật ký cảnh báo |
| 📷 Camera | **Video trực tiếp** qua WebRTC (mặc định). Nếu mạng chặn WebRTC thì dùng **Tự chụp mỗi giây**. Bảng trạng thái, ảnh cảnh báo gần nhất và nhật ký tự cập nhật |

### Trang Hiệu năng mô hình

Số liệu thật của lần huấn luyện và đánh giá (`assets/mo_hinh/`, số theo lớp lấy từ kết quả `model.val()` trong notebook): Precision, Recall, mAP50, mAP50-95 trên tập val / test; so sánh mô hình mới (v2, huấn luyện lại sau khi thêm 91 ảnh tự gán nhãn) với mô hình cũ (v1), cả về chỉ số nhận diện lẫn kết luận tuân thủ của từng công nhân; biểu đồ từng lớp; bản đồ nhiệt lớp × chỉ số; phân bố dữ liệu; đường cong huấn luyện 50 epoch; và các biểu đồ gốc của Ultralytics (ma trận nhầm lẫn, đường PR / F1 / P / R, results.png).

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

# ⛑️ Hệ thống giám sát an toàn công trường (bản web)

Đồ án môn Học máy (Nhóm 1). Mô hình **YOLOv11n** được tinh chỉnh trên bộ dữ liệu PPE để kiểm tra **từng công nhân** có đủ 4 trang bị bắt buộc hay không: **mũ bảo hộ, găng tay, áo bảo hộ, giày bảo hộ**. Ứng dụng cũng cảnh báo khi phát hiện công nhân để tay trần.

Bản web này chạy trên [Streamlit Community Cloud](https://streamlit.io/cloud), chỉ dùng CPU.

## Chức năng

| Tab | Mô tả |
|---|---|
| 🖼️ Ảnh | Tải một hoặc nhiều ảnh (hoặc chọn ảnh mẫu). Kết quả gồm ảnh đã đánh dấu, trạng thái AN TOÀN / CẢNH BÁO, bảng ✔/✘ cho từng công nhân, nút tải ảnh kết quả và nhật ký CSV |
| 🎞️ Video | Phân tích 30 giây đầu của video. Kết quả gồm video đã đánh dấu, ảnh các khoảnh khắc vi phạm và nhật ký cảnh báo |
| 📷 Camera | Chụp ảnh từ webcam của trình duyệt và phân tích ngay |

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

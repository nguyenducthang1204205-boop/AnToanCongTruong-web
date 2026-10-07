# === CELL 1 ===
from pathlib import Path
import csv
import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import cv2
from ultralytics import YOLO

PROJECT_DIR = Path.cwd()
DATASET_DIR = PROJECT_DIR / "dataset"
DATA_YAML = DATASET_DIR / "data.yaml"
WEIGHTS_PATH = PROJECT_DIR / "yolo11n.pt"                 # trọng số gốc (pretrained COCO)
RUNS_DIR = PROJECT_DIR / "runs" / "detect"
BEST_WEIGHTS = RUNS_DIR / "train" / "weights" / "best.pt"
LAST_WEIGHTS = RUNS_DIR / "train" / "weights" / "last.pt"
INPUT_DIR = PROJECT_DIR / "input_PPE"
OUTPUT_DIR = PROJECT_DIR / "output_PPE"
ALERT_DIR = OUTPUT_DIR / "canh_bao"                       # ảnh chụp khoảnh khắc vi phạm
LOG_PATH = OUTPUT_DIR / "nhat_ky_vi_pham.csv"             # nhật ký cảnh báo
for folder in (INPUT_DIR, OUTPUT_DIR, ALERT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

CONF_THRESHOLD = 0.3
PERSON_LABEL = "Person"

# 4 trang bị bắt buộc: (tên tiếng Việt, tên không dấu để vẽ lên ảnh, lớp "có", lớp "không có").
# cv2.putText không hiển thị được tiếng Việt có dấu nên chữ trên ảnh viết không dấu.
PPE_RULES = [
    ("Mũ bảo hộ", "Mu", "Helmet", "no-helmet"),
    ("Găng tay", "Gang tay", "Gloves", None),
    ("Áo bảo hộ", "Ao", "Safety Vest", "no-vest"),
    ("Giày bảo hộ", "Giay", "Safety Boot", "no-boot"),
]
EXTRA_VIOLATIONS = {"bare-arms": ("Để tay trần", "Tay tran")}
DANGER_LABELS = {"no-helmet", "no-vest", "no-boot", "bare-arms"}

print(f"Project: {PROJECT_DIR}")
print(f"Dataset config: {DATA_YAML}")
print(f"Weights: {WEIGHTS_PATH}")

# === CELL 2 ===
import ultralytics
print(f"Ultralytics: {ultralytics.__version__}")

required_paths = [
    DATA_YAML,
    DATASET_DIR / "train" / "images",
    DATASET_DIR / "train" / "labels",
    DATASET_DIR / "valid" / "images",
    DATASET_DIR / "valid" / "labels",
    DATASET_DIR / "test" / "images",
    DATASET_DIR / "test" / "labels",
]
for path in required_paths:
    print(f"{'OK' if path.exists() else 'MISSING'}: {path}")

# === CELL 3 ===
RETRAIN = False

if BEST_WEIGHTS.exists() and not RETRAIN:
    print(f"Đã có trọng số huấn luyện, bỏ qua bước train: {BEST_WEIGHTS}")
else:
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Chưa tìm thấy {WEIGHTS_PATH}. Hãy đặt yolo11n.pt vào thư mục dự án "
            "hoặc dùng đường dẫn trọng số khác."
        )
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Chưa tìm thấy file cấu hình: {DATA_YAML}")

    model_v11 = YOLO(str(WEIGHTS_PATH))
    train_results = model_v11.train(
        data=str(DATA_YAML),
        task="detect",
        imgsz=640,
        epochs=50,
        project=str(RUNS_DIR),
        name="train",
        exist_ok=True,
        device="0",
    )
    print(f"Best weights: {BEST_WEIGHTS}")
    print(f"Last weights: {LAST_WEIGHTS}")

# === CELL 4 ===
eval_model = YOLO(str(BEST_WEIGHTS))
ppe_classes = {have for _, _, have, _ in PPE_RULES}

for split in ("val", "test"):
    metrics = eval_model.val(
        data=str(DATA_YAML),
        split=split,
        imgsz=640,
        device="0",
        project=str(RUNS_DIR),
        name=f"eval_{split}",
        exist_ok=True,
        workers=0,   # tránh treo khi chạy dạng script trên Windows (DataLoader đa tiến trình)
        verbose=False,
    )
    box = metrics.box
    print(f"\n=== Tập {split}: P = {box.mp:.3f}, R = {box.mr:.3f}, "
          f"mAP50 = {box.map50:.3f}, mAP50-95 = {box.map:.3f}")
    print(f"    Tốc độ suy luận: {metrics.speed['inference']:.1f} ms/ảnh "
          f"(~{1000 / sum(metrics.speed.values()):.0f} FPS tính cả tiền/hậu xử lý)")
    print(f"    {'Lớp':<13}{'P':>7}{'R':>7}{'mAP50':>8}{'mAP50-95':>10}")
    for i, class_id in enumerate(box.ap_class_index):
        name = eval_model.names[int(class_id)]
        mark = "  <- trang bị bắt buộc" if name in ppe_classes else ""
        print(f"    {name:<13}{box.p[i]:>7.3f}{box.r[i]:>7.3f}"
              f"{box.ap50[i]:>8.3f}{box.ap[i]:>10.3f}{mark}")

# === CELL 5 ===
detector = YOLO(str(BEST_WEIGHTS if BEST_WEIGHTS.exists() else WEIGHTS_PATH))
print(f"Mô hình giám sát: {BEST_WEIGHTS if BEST_WEIGHTS.exists() else WEIGHTS_PATH}")

GREEN, RED, BLUE, WHITE = (0, 170, 0), (0, 0, 255), (255, 170, 0), (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def tim_cong_nhan(item_box, workers):
    """Trả về công nhân sở hữu trang bị (box người chứa tâm trang bị), hoặc None."""
    x1, y1, x2, y2 = item_box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    owner, best_score = None, float("inf")
    for worker in workers:
        px1, py1, px2, py2 = worker["xyxy"]
        mx, my = 0.1 * (px2 - px1), 0.1 * (py2 - py1)
        if px1 - mx <= cx <= px2 + mx and py1 - my <= cy <= py2 + my:
            score = abs(cx - (px1 + px2) / 2) / max(px2 - px1, 1)
            if score < best_score:
                owner, best_score = worker, score
    return owner


def phan_tich_khung_hinh(model, frame, conf=CONF_THRESHOLD):
    """Nhận diện và kiểm tra từng công nhân có đủ 4 trang bị bảo hộ hay không."""
    result = model.predict(source=frame, conf=conf, verbose=False)[0]
    persons, items = [], []
    for box in result.boxes:
        det = {
            "label": model.names[int(box.cls.item())],
            "conf": float(box.conf.item()),
            "xyxy": tuple(map(int, box.xyxy[0].tolist())),
        }
        (persons if det["label"] == PERSON_LABEL else items).append(det)

    persons.sort(key=lambda d: d["xyxy"][0])  # đánh số công nhân từ trái sang phải
    workers = [{"id": i, "xyxy": p["xyxy"], "items": []} for i, p in enumerate(persons, 1)]
    unassigned = []
    for item in items:
        owner = tim_cong_nhan(item["xyxy"], workers)
        (owner["items"] if owner else unassigned).append(item)

    for worker in workers:
        labels = {d["label"] for d in worker["items"]}
        worker["thieu"] = [(name, short) for name, short, have, lack in PPE_RULES
                           if have not in labels or lack in labels]
        worker["thieu"] += [EXTRA_VIOLATIONS[l] for l in EXTRA_VIOLATIONS if l in labels]
        worker["dat_chuan"] = not worker["thieu"]

    orphans = [d for d in unassigned if d["label"] in DANGER_LABELS]
    so_nguoi_vi_pham = sum(not w["dat_chuan"] for w in workers)
    return {
        "workers": workers,
        "items": items,
        "orphan_violations": orphans,
        "so_cong_nhan": len(workers),
        "so_nguoi_vi_pham": so_nguoi_vi_pham,
        "canh_bao": so_nguoi_vi_pham > 0 or bool(orphans),
    }


def ghi_nhan(img, text, x, y, color, scale, placed=None, min_y=0):
    """Viết chữ trắng trên nền màu, đáy chữ tại (x, y). Nếu truyền `placed` (các nhãn đã vẽ),
    nhãn được dời xuống cho đến khi không đè lên nhãn khác."""
    thickness = max(1, round(scale * 2))
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thickness)
    x = min(max(0, x), max(0, img.shape[1] - tw))
    y = max(y, min_y + th + base)
    if placed is not None:
        def overlaps(top):
            return any(x < px2 and px1 < x + tw and top < py2 and py1 < top + th + base
                       for px1, py1, px2, py2 in placed)
        while overlaps(y - th - base) and y < img.shape[0]:
            y += th + base + 2
        placed.append((x, y - th - base, x + tw, y))
    cv2.rectangle(img, (x, y - th - base), (x + tw, y), color, -1)
    cv2.putText(img, text, (x, y - base), FONT, scale, WHITE, thickness)


def ve_ket_qua(frame, report, fps=None, ve_trang_bi=True):
    """Vẽ box công nhân (xanh = đủ, đỏ = thiếu), box trang bị, banner cảnh báo và FPS."""
    img = frame.copy()
    h, w = img.shape[:2]
    scale = max(0.45, min(w, h) / 1400)
    bar_h = int(60 * scale)

    if ve_trang_bi:
        for d in report["items"]:
            x1, y1, x2, y2 = d["xyxy"]
            cv2.rectangle(img, (x1, y1), (x2, y2), RED if d["label"] in DANGER_LABELS else BLUE, 1)

    labels = []
    for worker in report["workers"]:
        x1, y1, x2, y2 = worker["xyxy"]
        color = GREEN if worker["dat_chuan"] else RED
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        detail = "OK" if worker["dat_chuan"] else ", ".join(short for _, short in worker["thieu"])
        labels.append((f"#{worker['id']}: {detail}", x1, y1 - 2, color, scale))

    for d in report["orphan_violations"]:
        x1, y1, x2, y2 = d["xyxy"]
        cv2.rectangle(img, (x1, y1), (x2, y2), RED, 2)
        labels.append((d["label"], x1, y1 - 2, RED, scale * 0.8))

    placed = []
    for text, x, y, color, s in labels:
        ghi_nhan(img, text, x, y, color, s, placed=placed, min_y=bar_h + 2)

    n, k = report["so_cong_nhan"], report["so_nguoi_vi_pham"]
    if report["canh_bao"]:
        banner, color = (f"CANH BAO: {k}/{n} cong nhan thieu do bao ho" if k
                         else "CANH BAO: phat hien vi pham bao ho"), RED
    elif n:
        banner, color = f"AN TOAN: {n}/{n} cong nhan du do bao ho", GREEN
    else:
        banner, color = "Khong phat hien cong nhan", (90, 90, 90)
    cv2.rectangle(img, (0, 0), (w, bar_h), color, -1)
    cv2.putText(img, banner, (10, int(bar_h * 0.72)), FONT, scale * 1.2, WHITE, max(1, round(scale * 2.5)))
    if fps is not None:
        ghi_nhan(img, f"FPS: {fps:.1f}", 10, h - 10, (40, 40, 40), scale)
    return img


def mo_ta_vi_pham(report):
    parts = [f"#{w['id']} thiếu {', '.join(name for name, _ in w['thieu'])}"
             for w in report["workers"] if not w["dat_chuan"]]
    parts += [f"{d['label']} (chưa xác định công nhân)" for d in report["orphan_violations"]]
    return "; ".join(parts)


def in_bao_cao(nguon, report):
    print(f"[{nguon}] {report['so_cong_nhan']} công nhân, {report['so_nguoi_vi_pham']} người vi phạm")
    for w in report["workers"]:
        status = "ĐẦY ĐỦ" if w["dat_chuan"] else "THIẾU " + ", ".join(name for name, _ in w["thieu"])
        print(f"    Công nhân #{w['id']}: {status}")
    for d in report["orphan_violations"]:
        print(f"    Vi phạm chưa xác định công nhân: {d['label']} ({d['conf']:.2f})")


def ghi_nhat_ky(nguon, thoi_diem, report, anh_canh_bao):
    """Ghi mỗi công nhân vi phạm thành một dòng trong file CSV (mở được bằng Excel)."""
    is_new = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
        log = csv.writer(f)
        if is_new:
            log.writerow(["thoi_gian", "nguon", "thoi_diem_giay", "cong_nhan", "vi_pham", "anh_canh_bao"])
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for w in report["workers"]:
            if not w["dat_chuan"]:
                log.writerow([now, nguon, thoi_diem, f"#{w['id']}",
                              ", ".join(name for name, _ in w["thieu"]), anh_canh_bao])
        for d in report["orphan_violations"]:
            log.writerow([now, nguon, thoi_diem, "không xác định", d["label"], anh_canh_bao])


def giam_sat_video(source, output_path=None, hien_thi=False, chu_ky_canh_bao=3.0):
    """Giám sát video hoặc camera. Khi có vi phạm: in cảnh báo, lưu ảnh và ghi nhật ký
    (tối đa một lần mỗi `chu_ky_canh_bao` giây để không bị lặp)."""
    is_camera = isinstance(source, int) or str(source).lower().startswith(("rtsp://", "http://", "https://"))
    ten_nguon = f"camera {source}" if is_camera else Path(source).name
    ten_file = "camera" if is_camera else Path(source).stem

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Không thể mở nguồn video: {source}")
    src_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    frame_count, violation_frames, fps = 0, 0, 0.0
    last_alert = -chu_ky_canh_bao
    start = time.perf_counter()
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            t0 = time.perf_counter()
            report = phan_tich_khung_hinh(detector, frame)
            dt = max(time.perf_counter() - t0, 1e-6)
            fps = 1 / dt if fps == 0 else 0.9 * fps + 0.1 / dt
            annotated = ve_ket_qua(frame, report, fps=fps)

            thoi_diem = time.perf_counter() - start if is_camera else frame_count / src_fps
            if report["canh_bao"]:
                violation_frames += 1
                if thoi_diem - last_alert >= chu_ky_canh_bao:
                    last_alert = thoi_diem
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S") if is_camera else f"{thoi_diem:06.1f}s"
                    alert_path = ALERT_DIR / f"{ten_file}_{stamp}.jpg"
                    cv2.imwrite(str(alert_path), annotated)
                    ghi_nhat_ky(ten_nguon, f"{thoi_diem:.1f}", report, alert_path.name)
                    print(f"[CẢNH BÁO {thoi_diem:6.1f}s] {mo_ta_vi_pham(report)}")

            if output_path is not None:
                if writer is None:
                    h, w = annotated.shape[:2]
                    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
                writer.write(annotated)
            if hien_thi:
                cv2.imshow("Giam sat an toan cong truong - nhan Q de thoat", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            frame_count += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if hien_thi:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start
    ratio = violation_frames / frame_count if frame_count else 0
    print(f"Đã xử lý {frame_count} frame từ {ten_nguon} "
          f"({frame_count / elapsed if elapsed else 0:.1f} FPS trung bình), "
          f"{violation_frames} frame có vi phạm ({ratio:.0%}).")
    if output_path is not None:
        print(f"Video kết quả: {output_path}")
    return {"frames": frame_count, "violation_frames": violation_frames}

# === CELL 6 ===
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
image_paths = sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS)

if not image_paths:
    print(f"Chưa có ảnh trong {INPUT_DIR}")
for image_path in image_paths:
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Không đọc được ảnh: {image_path}")
        continue
    report = phan_tich_khung_hinh(detector, frame)
    output_path = OUTPUT_DIR / f"{image_path.stem}_giam_sat.jpg"
    cv2.imwrite(str(output_path), ve_ket_qua(frame, report))
    in_bao_cao(image_path.name, report)
    if report["canh_bao"]:
        ghi_nhat_ky(image_path.name, "", report, output_path.name)
print(f"\nẢnh kết quả lưu trong: {OUTPUT_DIR}")

# === CELL 7 ===
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
video_paths = sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in VIDEO_EXTS)

if not video_paths:
    print(f"Chưa có video đầu vào trong {INPUT_DIR}")
for video_path in video_paths:
    giam_sat_video(str(video_path), output_path=OUTPUT_DIR / f"{video_path.stem}_giam_sat.mp4")

# === CELL 8 ===
CAMERA_SOURCE = 0
RUN_CAMERA = False

if RUN_CAMERA:
    giam_sat_video(CAMERA_SOURCE, output_path=OUTPUT_DIR / "camera_giam_sat.mp4", hien_thi=True)
else:
    print("Bỏ qua camera. Đặt RUN_CAMERA = True để giám sát thời gian thực.")

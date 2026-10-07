"""Bản web của hệ thống giám sát an toàn công trường (Streamlit Community Cloud, chạy bằng CPU).

Dùng lại nguyên logic của notebook: nạp cell 1 (cấu hình) và cell 5 (mô hình + kiểm tra trang bị)
từ notebook_code.py, giống bản desktop giao_dien.py.

Chạy thử trên máy (từ thư mục dự án):
    .venv\\Scripts\\streamlit.exe run streamlit_app\\streamlit_app.py
"""
import csv
import io
import os
import subprocess
import tempfile
import threading
import types
from datetime import datetime
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
os.chdir(APP_DIR)  # notebook dùng PROJECT_DIR = Path.cwd(); trọng số ở runs/detect/train/weights/best.pt

GIAY_VIDEO_TOI_DA = 30   # máy chủ miễn phí chỉ có CPU yếu nên giới hạn độ dài video
CANH_TOI_DA = 960        # thu nhỏ khung hình video lớn hơn mức này
SO_ANH_CANH_BAO_TOI_DA = 12
COT_CONG_NHAN = ["CN", "Mũ", "Găng", "Áo", "Giày", "Tay", "Kết luận"]
COT_NHAT_KY = ["Thời gian", "Nguồn", "Thời điểm (s)", "Công nhân", "Vi phạm", "Ảnh cảnh báo"]

st.set_page_config(page_title="Giám sát an toàn công trường", page_icon="⛑️", layout="wide")


@st.cache_resource(show_spinner="Đang nạp mô hình YOLOv11…")
def nap_logic_notebook():
    """Chạy cell 1 (cấu hình) và cell 5 (mô hình + logic tuân thủ) của notebook_code.py, một lần cho mọi phiên."""
    code = (APP_DIR / "notebook_code.py").read_text(encoding="utf-8")
    cells = {}
    for part in code.split("# === CELL ")[1:]:
        so, _, body = part.partition(" ===\n")
        cells[int(so)] = body
    core = types.ModuleType("giam_sat_core")
    core.__file__ = str(APP_DIR / "notebook_code.py")
    for so in (1, 5):
        exec(compile(cells[so], f"notebook_code.py (cell {so})", "exec"), core.__dict__)
    return core, threading.Lock()  # khóa: nhiều phiên dùng chung một mô hình


core, khoa_mo_hinh = nap_logic_notebook()


def phan_tich(frame, conf, ve=True):
    with khoa_mo_hinh:
        report = core.phan_tich_khung_hinh(core.detector, frame, conf=conf)
    return report, (core.ve_ket_qua(frame, report) if ve else None)


# ---------------------------------------------------------------------- trình bày kết quả
def the_trang_thai(mau, tieu_de, mo_ta):
    nen = {"do": "#dc2626", "xanh": "#16a34a", "xam": "#64748b"}[mau]
    st.markdown(f'<div style="background:{nen};color:#fff;border-radius:14px;padding:14px 18px;'
                f'text-align:center;margin-bottom:12px">'
                f'<div style="font-size:1.5em;font-weight:700;letter-spacing:.5px">{tieu_de}</div>'
                f'<div style="opacity:.95">{mo_ta}</div></div>', unsafe_allow_html=True)


def the_tu_bao_cao(report):
    n, k = report["so_cong_nhan"], report["so_nguoi_vi_pham"]
    if report["canh_bao"]:
        the_trang_thai("do", "⚠ CẢNH BÁO", f"{k}/{n} công nhân thiếu đồ bảo hộ" if k else "Phát hiện vi phạm bảo hộ")
    elif n:
        the_trang_thai("xanh", "✔ AN TOÀN", f"{n}/{n} công nhân đủ đồ bảo hộ")
    else:
        the_trang_thai("xam", "KHÔNG CÓ CÔNG NHÂN", "Không phát hiện người trong ảnh")


def bang_cong_nhan(report):
    rows = []
    for w in report["workers"]:
        thieu = {name for name, _ in w["thieu"]}
        dau = ["✘" if name in thieu else "✔" for name, _, _, _ in core.PPE_RULES]
        dau.append("✘" if any(name in thieu for name, _ in core.EXTRA_VIOLATIONS.values()) else "✔")
        rows.append([f"#{w['id']}", *dau, "Đạt" if w["dat_chuan"] else "Vi phạm"])
    for d in report["orphan_violations"]:
        rows.append(["?", "", "", "", "", "", f"{d['label']} (chưa gắn được công nhân)"])
    return pd.DataFrame(rows, columns=COT_CONG_NHAN)


def to_mau_bang(df):
    """Tô chữ đỏ cho ô ✘ / Vi phạm, xanh cho ô ✔ / Đạt."""
    def mau(v):
        if v in ("✘", "Vi phạm"):
            return "color:#dc2626;font-weight:600"
        if v in ("✔", "Đạt"):
            return "color:#16a34a"
        return ""
    return df.style.map(mau)


def dong_nhat_ky(nguon, thoi_diem, report, ten_anh):
    """Các dòng nhật ký cùng định dạng với ghi_nhat_ky của notebook."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [[now, nguon, thoi_diem, f"#{w['id']}", ", ".join(name for name, _ in w["thieu"]), ten_anh]
            for w in report["workers"] if not w["dat_chuan"]]
    rows += [[now, nguon, thoi_diem, "không xác định", d["label"], ten_anh] for d in report["orphan_violations"]]
    return rows


def csv_nhat_ky(rows):
    buf = io.StringIO()
    log = csv.writer(buf)
    log.writerow(["thoi_gian", "nguon", "thoi_diem_giay", "cong_nhan", "vi_pham", "anh_canh_bao"])
    log.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")  # utf-8-sig để Excel hiển thị đúng tiếng Việt


def gioi_han_kich_thuoc(frame, canh=CANH_TOI_DA):
    h, w = frame.shape[:2]
    ti_le = canh / max(h, w)
    if ti_le >= 1:
        return frame
    return cv2.resize(frame, (int(w * ti_le), int(h * ti_le)), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------- ảnh và camera
@st.cache_data(max_entries=64, show_spinner=False)
def phan_tich_anh(du_lieu: bytes, conf: float):
    """Kết quả được nhớ theo (ảnh, ngưỡng) nên bấm nút khác trên trang không phải chạy lại mô hình."""
    bgr = cv2.imdecode(np.frombuffer(du_lieu, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return None, None
    report, annotated = phan_tich(bgr, conf)
    return report, cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


def hien_thi_ket_qua_anh(ten, du_lieu, conf, khoa, xep_doc=False):
    report, jpg = phan_tich_anh(du_lieu, conf)
    if report is None:
        st.error(f"Không đọc được ảnh: {ten}")
        return
    trai, phai = (st.container(), st.container()) if xep_doc else st.columns([3, 2], gap="medium")
    with trai:
        st.image(jpg, caption=ten, width="stretch")
    with phai:
        the_tu_bao_cao(report)
        st.dataframe(to_mau_bang(bang_cong_nhan(report)), hide_index=True, width="stretch")
        ten_goc = Path(ten).stem
        nut1, nut2 = st.columns(2)
        nut1.download_button("⬇ Ảnh kết quả", jpg, f"{ten_goc}_giam_sat.jpg", "image/jpeg",
                             key=f"anh_{khoa}", width="stretch")
        rows = dong_nhat_ky(ten, "", report, f"{ten_goc}_giam_sat.jpg")
        nut2.download_button("⬇ Nhật ký CSV", csv_nhat_ky(rows), f"{ten_goc}_nhat_ky.csv", "text/csv",
                             key=f"csv_{khoa}", disabled=not rows, width="stretch")


# ---------------------------------------------------------------------- video
def sang_h264(nguon, dich):
    """Trình duyệt không phát được mp4v của OpenCV, nên chuyển sang H.264 bằng ffmpeg."""
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(nguon),
                    "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(dich)], check=True)


def phan_tich_video(du_lieu, ten, conf, chu_ky, khung_moi_giay, thanh_tien_do):
    """Giống giam_sat_video của notebook. Để kịp trên CPU, chỉ chạy mô hình `khung_moi_giay` lần mỗi giây;
    các khung ở giữa vẽ lại kết quả gần nhất nên video vẫn mượt."""
    thu_muc = Path(tempfile.mkdtemp())
    vao, tam, ra = thu_muc / f"vao{Path(ten).suffix or '.mp4'}", thu_muc / "tam.mp4", thu_muc / "ra.mp4"
    vao.write_bytes(du_lieu)

    cap = cv2.VideoCapture(str(vao))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps > 120:
        fps = 30.0
    tong = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    so_khung = min(tong, int(fps * GIAY_VIDEO_TOI_DA)) if tong > 0 else int(fps * GIAY_VIDEO_TOI_DA)
    buoc = max(1, round(fps / khung_moi_giay))

    writer, report = None, None
    frames, vp_frames, so_lan_chay, last_alert = 0, 0, 0, float("-inf")
    canh_bao, rows = [], []
    try:
        while frames < so_khung:
            ok, frame = cap.read()
            if not ok:
                break
            frame = gioi_han_kich_thuoc(frame)
            thoi_diem = frames / fps
            if frames % buoc == 0:
                report, annotated = phan_tich(frame, conf)
                so_lan_chay += 1
                if report["canh_bao"] and thoi_diem - last_alert >= chu_ky:
                    last_alert = thoi_diem
                    ten_anh = f"canh_bao_{thoi_diem:06.1f}s.jpg"
                    rows += dong_nhat_ky(ten, f"{thoi_diem:.1f}", report, ten_anh)
                    if len(canh_bao) < SO_ANH_CANH_BAO_TOI_DA:
                        jpg = cv2.imencode(".jpg", gioi_han_kich_thuoc(annotated, 640))[1].tobytes()
                        canh_bao.append((jpg, f"{thoi_diem:.1f}s · {core.mo_ta_vi_pham(report)}"))
            else:
                annotated = core.ve_ket_qua(frame, report)
            vp_frames += report["canh_bao"]
            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(str(tam), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(annotated)
            frames += 1
            if frames % 10 == 0:
                thanh_tien_do.progress(min(frames / so_khung, 1.0) * 0.95,
                                       text=f"Đang phân tích… {thoi_diem:.1f}s / {so_khung / fps:.1f}s")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if frames == 0:
        return None
    thanh_tien_do.progress(0.97, text="Đang xuất video kết quả…")
    sang_h264(tam, ra)
    return {"video": ra.read_bytes(), "ten": ten, "frames": frames, "vp_frames": vp_frames, "tong": tong,
            "fps": fps, "so_lan_chay": so_lan_chay, "canh_bao": canh_bao, "rows": rows}


# ---------------------------------------------------------------------- giao diện
st.markdown("""
<style>
  .block-container {padding-top: 3.5rem; max-width: 1300px}
  [data-testid="stImageCaption"] {font-size: .85rem}
</style>
<div style="display:flex;align-items:center;gap:14px;margin-bottom:4px">
  <div style="font-size:44px;line-height:1">⛑️</div>
  <div>
    <div style="font-size:1.7em;font-weight:700">Hệ thống giám sát an toàn công trường</div>
    <div style="opacity:.7">YOLOv11n · kiểm tra <b>Mũ</b>, <b>Găng tay</b>, <b>Áo</b> và <b>Giày</b> bảo hộ cho từng công nhân</div>
  </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Cài đặt")
    conf = st.slider("Ngưỡng tin cậy", 0.10, 0.90, float(core.CONF_THRESHOLD), 0.05,
                     help="Thấp: phát hiện nhiều hơn nhưng dễ nhầm. Cao: chắc chắn hơn nhưng dễ bỏ sót.")
    chu_ky = st.slider("Chu kỳ cảnh báo video (giây)", 1, 10, 3,
                       help="Khoảng cách tối thiểu giữa hai lần ghi cảnh báo trong video.")
    khung_moi_giay = st.slider("Số lần phân tích mỗi giây video", 1, 10, 4,
                               help="Càng cao càng chính xác nhưng càng lâu (máy chủ miễn phí chỉ có CPU).")
    st.divider()
    st.markdown("**Cách đọc kết quả**\n\n"
                "- Khung **xanh**: công nhân đủ 4 trang bị\n"
                "- Khung **đỏ**: thiếu trang bị (ghi rõ món thiếu)\n"
                "- Bảng: ✔ có · ✘ thiếu · *Tay ✘* = để tay trần")
    st.caption("Mô hình YOLOv11n tinh chỉnh trên bộ dữ liệu PPE (Roboflow, CC BY 4.0). "
               "Không có lớp *no-gloves* nên thiếu găng tay được suy ra từ việc không phát hiện găng tay. "
               "Ảnh và video chỉ được xử lý tạm thời, không lưu lại.")

tab_anh, tab_video, tab_camera = st.tabs(["🖼️  Ảnh", "🎞️  Video", "📷  Camera"])

with tab_anh:
    nguon = st.segmented_control("Nguồn ảnh", ["Tải ảnh lên", "Ảnh mẫu"], default="Tải ảnh lên",
                                 label_visibility="collapsed")
    if nguon == "Ảnh mẫu":
        mau = sorted((APP_DIR / "vi_du").glob("*.jpg"))
        chon = st.pills("Chọn ảnh mẫu", [p.name for p in mau], default=mau[0].name if mau else None)
        anh = [(chon, (APP_DIR / "vi_du" / chon).read_bytes())] if chon else []
    else:
        tep = st.file_uploader("Kéo thả hoặc chọn một / nhiều ảnh công trường", type=["jpg", "jpeg", "png", "bmp", "webp"],
                               accept_multiple_files=True)
        anh = [(f.name, f.getvalue()) for f in tep or []]
    if not anh:
        st.info("Chọn ảnh để bắt đầu. Có thể chọn nhiều ảnh cùng lúc.")
    for i, (ten, du_lieu) in enumerate(anh):
        with st.container(border=True):
            with st.spinner(f"Đang phân tích {ten}…"):
                hien_thi_ket_qua_anh(ten, du_lieu, conf, f"a{i}")

with tab_video:
    tep_video = st.file_uploader("Chọn video công trường", type=["mp4", "avi", "mov", "mkv"])
    st.caption(f"Máy chủ miễn phí chỉ có CPU nên chỉ phân tích **{GIAY_VIDEO_TOI_DA} giây đầu** của video. "
               "Video 30 giây mất khoảng 1–3 phút.")
    if tep_video is not None and st.button("▶ Phân tích video", type="primary"):
        thanh = st.progress(0.0, text="Đang chuẩn bị…")
        kq = phan_tich_video(tep_video.getvalue(), tep_video.name, conf, chu_ky, khung_moi_giay, thanh)
        thanh.empty()
        st.session_state["video_kq"] = kq
        if kq is None:
            st.error("Không đọc được video. Hãy thử file .mp4 khác.")

    kq = st.session_state.get("video_kq")
    if kq and tep_video is not None and kq["ten"] == tep_video.name:
        trai, phai = st.columns([3, 2], gap="medium")
        with trai:
            st.video(kq["video"])
        with phai:
            mo_ta = (f"{kq['vp_frames']}/{kq['frames']} khung hình có vi phạm "
                     f"({kq['vp_frames'] / kq['frames']:.0%}) · {len({r[2] for r in kq['rows']})} lần cảnh báo")
            if kq["vp_frames"]:
                the_trang_thai("do", "⚠ CÓ VI PHẠM", mo_ta)
            else:
                the_trang_thai("xanh", "✔ AN TOÀN", mo_ta)
            if kq["tong"] > kq["frames"]:
                st.caption(f"Đã phân tích {kq['frames'] / kq['fps']:.0f} giây đầu trong tổng "
                           f"{kq['tong'] / kq['fps']:.0f} giây ({kq['so_lan_chay']} lần chạy mô hình).")
            ten_goc = Path(kq["ten"]).stem
            st.download_button("⬇ Tải video kết quả", kq["video"], f"{ten_goc}_giam_sat.mp4", "video/mp4",
                               width="stretch")
            st.download_button("⬇ Tải nhật ký CSV", csv_nhat_ky(kq["rows"]), f"{ten_goc}_nhat_ky.csv", "text/csv",
                               disabled=not kq["rows"], width="stretch")
        if kq["canh_bao"]:
            st.subheader("Khoảnh khắc vi phạm")
            cot = st.columns(4)
            for i, (jpg, chu_thich) in enumerate(kq["canh_bao"]):
                cot[i % 4].image(jpg, caption=chu_thich, width="stretch")
            st.subheader("Nhật ký cảnh báo")
            st.dataframe(pd.DataFrame(kq["rows"], columns=COT_NHAT_KY), hide_index=True, width="stretch")

with tab_camera:
    st.caption("Bấm **Take Photo** để chụp từ webcam, ảnh sẽ được phân tích ngay. "
               "Trình duyệt sẽ hỏi quyền dùng camera ở lần đầu.")
    trai, phai = st.columns([2, 3], gap="medium")
    with trai:
        anh_cam = st.camera_input("Webcam", label_visibility="collapsed")
    with phai:
        if anh_cam is None:
            st.info("Chưa có ảnh chụp.")
        else:
            hien_thi_ket_qua_anh("webcam.jpg", anh_cam.getvalue(), conf, "cam", xep_doc=True)

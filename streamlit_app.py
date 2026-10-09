"""Bản web của hệ thống giám sát an toàn công trường (Streamlit Community Cloud, chạy bằng CPU).

Dùng lại nguyên logic của notebook: nạp cell 1 (cấu hình) và cell 5 (mô hình + kiểm tra trang bị)
từ notebook_code.py, giống bản desktop giao_dien.py.

Chạy thử trên máy (từ thư mục dự án):
    .venv\\Scripts\\streamlit.exe run streamlit_app\\streamlit_app.py
"""
import base64
import collections
import csv
import io
import os
import shutil
import subprocess
import tempfile
import threading
import time
import types
from datetime import datetime
from pathlib import Path

import av
import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

import bieu_do as bd

APP_DIR = Path(__file__).resolve().parent
os.chdir(APP_DIR)  # notebook dùng PROJECT_DIR = Path.cwd(); trọng số ở runs/detect/train/weights/best.pt

CANH_TOI_DA = 960        # thu nhỏ khung hình video lớn hơn mức này
SO_ANH_CANH_BAO_TOI_DA = 12
COT_CONG_NHAN = ["CN", "Mũ", "Găng", "Áo", "Giày", "Kết luận"]
COT_NHAT_KY = ["Thời gian", "Nguồn", "Thời điểm (s)", "Công nhân", "Vi phạm", "Ảnh cảnh báo"]

st.set_page_config(page_title="Giám sát an toàn công trường", page_icon=":material/engineering:", layout="wide")


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
    import torch
    if not torch.cuda.is_available():
        # Máy chủ miễn phí chỉ được khoảng 1–2 lõi nhưng PyTorch thấy toàn bộ lõi của máy thật và tạo quá nhiều
        # luồng; giới hạn lại để còn CPU cho bộ mã hóa video WebRTC.
        torch.set_num_threads(2)
    return core, threading.Lock()  # khóa: nhiều phiên dùng chung một mô hình


core, khoa_mo_hinh = nap_logic_notebook()


def phan_tich(frame, conf, ve=True):
    with khoa_mo_hinh:
        report = core.phan_tich_khung_hinh(core.detector, frame, conf=conf)
    return report, (core.ve_ket_qua(frame, report) if ve else None)


# ---------------------------------------------------------------------- trình bày kết quả
# Biểu tượng SVG (Lucide, 24×24, nét theo màu chữ). HTML tự viết không biết người xem đang dùng nền sáng hay tối
# (st.context.theme có thể sai lúc mới tải trang), nên chỉ dùng nền đặc + chữ trắng hoặc màu trong suốt.
def svg(duong_ve, co=24):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{co}" height="{co}" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
            f'aria-hidden="true">{duong_ve}</svg>')


ICON_MU = ('<path d="M10 10V5a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1v5"/><path d="M14 6a6 6 0 0 1 6 6v3"/>'
           '<path d="M4 15v-3a6 6 0 0 1 6-6"/><rect x="2" y="15" width="20" height="4" rx="1"/>')
ICON_CANH_BAO = ('<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/>'
                 '<path d="M12 9v4"/><path d="M12 17h.01"/>')
ICON_AN_TOAN = ('<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1'
                'c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>'
                '<path d="m9 12 2 2 4-4"/>')
ICON_KHONG_NGUOI = ('<path d="M2 21a8 8 0 0 1 13.29-6"/><circle cx="10" cy="8" r="5"/>'
                    '<path d="m17 17 5 5"/><path d="m22 17-5 5"/>')
TRANG_THAI = {  # nền đặc, chữ trắng đạt tương phản ≥ 4.5:1 trên cả giao diện sáng và tối
    "do": ("#B91C1C", ICON_CANH_BAO),
    "xanh": ("#15803D", ICON_AN_TOAN),
    "xam": ("#475569", ICON_KHONG_NGUOI),
}


def the_trang_thai(mau, tieu_de, mo_ta):
    nen, icon = TRANG_THAI[mau]
    st.markdown(f'<div class="trang-thai" role="status" style="background:{nen}">{svg(icon, 30)}'
                f'<div><div class="tieu-de">{tieu_de}</div><div class="mo-ta">{mo_ta}</div></div></div>',
                unsafe_allow_html=True)


def chi_so(*cap):
    """Một hàng ô số liệu (nhãn, giá trị). Dùng container ngang thay cho st.columns để trên điện thoại các ô
    vẫn nằm cùng hàng thay vì xếp chồng."""
    with st.container(horizontal=True, gap="small"):
        for nhan, gia_tri in cap:
            st.metric(nhan, gia_tri, border=True, width="stretch")


def the_tu_bao_cao(report):
    n, k = report["so_cong_nhan"], report["so_nguoi_vi_pham"]
    if report["canh_bao"]:
        the_trang_thai("do", "CẢNH BÁO", f"{k}/{n} công nhân thiếu đồ bảo hộ" if k else "Phát hiện vi phạm bảo hộ")
    elif n:
        the_trang_thai("xanh", "AN TOÀN", f"{n}/{n} công nhân đủ đồ bảo hộ")
    else:
        the_trang_thai("xam", "KHÔNG CÓ CÔNG NHÂN", "Không phát hiện người trong khung hình")
    chi_so(("Công nhân", n), ("Đạt chuẩn", n - k), ("Vi phạm", k))


def bang_cong_nhan(report):
    rows = []
    for w in report["workers"]:
        thieu = {name for name, _ in w["thieu"]}
        dau = ["✘" if name in thieu else "✔" for name, _, _, _ in core.PPE_RULES]
        if w["dat_chuan"]:
            ket_luan = "Đạt"
        elif "✘" in dau:
            ket_luan = "Vi phạm"
        else:  # đủ 4 trang bị nhưng vẫn vi phạm, ví dụ để tay trần
            ket_luan = f"Vi phạm ({', '.join(name.lower() for name in thieu)})"
        rows.append([f"#{w['id']}", *dau, ket_luan])
    for d in report["orphan_violations"]:
        rows.append(["?", "", "", "", "", f"{d['label']} (chưa gắn được công nhân)"])
    return pd.DataFrame(rows, columns=COT_CONG_NHAN)


def to_mau_bang(df):
    """Tô nền đỏ nhạt cho ô ✘ / Vi phạm, xanh nhạt cho ô ✔ / Đạt. Màu nền trong suốt và chữ giữ màu của giao diện
    nên đọc được cả khi nền sáng lẫn tối."""
    def mau(v):
        if v == "✘" or str(v).startswith("Vi phạm"):
            return "background-color:rgba(220,38,38,.22);font-weight:700"
        if v in ("✔", "Đạt"):
            return "background-color:rgba(22,163,74,.16)"
        return ""
    return df.style.map(mau)


# Cột ✔/✘ hẹp để cột "Kết luận" không bị cắt trong khung bên phải
CAU_HINH_COT_CONG_NHAN = {"CN": st.column_config.TextColumn(width=44),
                          **{ten: st.column_config.TextColumn(width=52) for ten in COT_CONG_NHAN[1:5]}}


def hien_bang_cong_nhan(report):
    df = bang_cong_nhan(report)
    if not df.empty:
        st.dataframe(to_mau_bang(df), hide_index=True, width="stretch", column_config=CAU_HINH_COT_CONG_NHAN)


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
        hien_bang_cong_nhan(report)
        ten_goc = Path(ten).stem
        nut1, nut2 = st.columns(2)
        nut1.download_button("Ảnh kết quả", jpg, f"{ten_goc}_giam_sat.jpg", "image/jpeg",
                             key=f"anh_{khoa}", icon=":material/download:", width="stretch")
        rows = dong_nhat_ky(ten, "", report, f"{ten_goc}_giam_sat.jpg")
        nut2.download_button("Nhật ký CSV", csv_nhat_ky(rows), f"{ten_goc}_nhat_ky.csv", "text/csv",
                             key=f"csv_{khoa}", disabled=not rows, icon=":material/table_view:",
                             width="stretch")


# ---------------------------------------------------------------------- video
def sang_h264(nguon, dich, preset="veryfast"):
    """Trình duyệt không phát được mp4v của OpenCV, nên chuyển sang H.264 bằng ffmpeg."""
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(nguon),
                    "-c:v", "libx264", "-preset", preset, "-pix_fmt", "yuv420p", "-an",
                    "-movflags", "+faststart", str(dich)], check=True)


MA_HOA_TRINH_DUYET = {"avc1", "h264", "x264", "vp80", "vp90", "vp09", "av01"}


@st.cache_data(max_entries=4, show_spinner="Đang chuẩn bị bản xem trước của video gốc…")
def ban_xem_truoc(file_id, duoi, _du_lieu):
    """Video gốc để xem trên trình duyệt. Trình duyệt chỉ phát được H.264/VP8/VP9/AV1, nên video khác
    (mp4v, .avi, .mkv, HEVC của iPhone…) được chuyển sang H.264. Nhớ theo file_id để không chuyển lại."""
    thu_muc = Path(tempfile.mkdtemp())
    vao = thu_muc / f"goc{duoi}"
    vao.write_bytes(_du_lieu)
    cap = cv2.VideoCapture(str(vao))
    ma_hoa = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode("ascii", "ignore").lower()
    cap.release()
    if duoi in (".mp4", ".mov", ".m4v", ".webm") and ma_hoa in MA_HOA_TRINH_DUYET:
        return _du_lieu
    ra = thu_muc / "xem_truoc.mp4"
    try:
        sang_h264(vao, ra, preset="ultrafast")
    except subprocess.CalledProcessError:
        return _du_lieu  # để trình duyệt tự thử
    return ra.read_bytes()


@st.cache_data(max_entries=8, show_spinner=False)
def thoi_luong_video(file_id, duoi, _du_lieu):
    """Độ dài video (giây) đọc từ số khung hình / fps; 0 nếu không đọc được."""
    thu_muc = Path(tempfile.mkdtemp())
    try:
        vao = thu_muc / f"goc{duoi}"
        vao.write_bytes(_du_lieu)
        cap = cv2.VideoCapture(str(vao))
        fps, tong = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return tong / fps if fps and 0 < fps <= 120 and tong > 0 else 0
    finally:
        shutil.rmtree(thu_muc, ignore_errors=True)


# Phát hai video cùng lúc: tìm thẻ <video> trong hai container (class st-key-…) và nối các sự kiện phát / dừng /
# tua / đổi tốc độ của video này sang video kia. Component v2 chạy ngay trong trang (không nằm trong iframe) nên
# truy cập được các st.video. Video gốc làm mốc: khi đang phát mà lệch quá 0,3 giây thì kéo video kết quả về khớp.
DONG_BO_VIDEO_JS = """
export default function(component) {
    const [lopGoc, lopKetQua] = component.data.lop;
    let goc = null, ketQua = null, huy = null;

    const noi = (nguon, dich, signal) => {
        const khop = () => {
            if (Math.abs(dich.currentTime - nguon.currentTime) > 0.1) dich.currentTime = nguon.currentTime;
        };
        nguon.addEventListener('play', () => { khop(); if (dich.paused) dich.play().catch(() => {}); }, { signal });
        nguon.addEventListener('pause', () => { if (!dich.paused && !nguon.ended) dich.pause(); }, { signal });
        nguon.addEventListener('seeking', khop, { signal });
        nguon.addEventListener('ratechange', () => {
            if (dich.playbackRate !== nguon.playbackRate) dich.playbackRate = nguon.playbackRate;
        }, { signal });
    };

    const kiemTra = () => {
        const a = document.querySelector(`.${lopGoc} video`);
        const b = document.querySelector(`.${lopKetQua} video`);
        if (!a || !b) return;
        if (a !== goc || b !== ketQua) {  // lần đầu, hoặc Streamlit vừa vẽ lại thẻ video
            if (huy) huy.abort();
            huy = new AbortController();
            goc = a; ketQua = b;
            noi(goc, ketQua, huy.signal);
            noi(ketQua, goc, huy.signal);
        } else if (!goc.paused && !ketQua.paused && Math.abs(goc.currentTime - ketQua.currentTime) > 0.3) {
            ketQua.currentTime = goc.currentTime;
        }
    };
    kiemTra();
    const hen = setInterval(kiemTra, 500);
    return () => { clearInterval(hen); if (huy) huy.abort(); };
}
"""


@st.cache_resource
def tao_dong_bo_video():
    return st.components.v2.component("dong_bo_video", html="<span></span>", js=DONG_BO_VIDEO_JS)


def dinh_dang_thoi_luong(giay):
    phut, giay = divmod(round(giay), 60)
    return f"{phut} phút {giay:02d} giây" if phut else f"{giay} giây"


def phan_tich_video(du_lieu, ten, conf, chu_ky, khung_moi_giay, thanh_tien_do):
    """Giống giam_sat_video của notebook. Phân tích toàn bộ video và ghi đủ mọi khung hình với cùng fps, nên video
    kết quả dài đúng bằng video gốc. Để kịp trên CPU, chỉ chạy mô hình `khung_moi_giay` lần mỗi giây;
    các khung ở giữa vẽ lại kết quả gần nhất nên video vẫn mượt."""
    thu_muc = Path(tempfile.mkdtemp())
    try:
        return _phan_tich_video(thu_muc, du_lieu, ten, conf, chu_ky, khung_moi_giay, thanh_tien_do)
    finally:
        shutil.rmtree(thu_muc, ignore_errors=True)  # video dài để lại file tạm lớn trên máy chủ


def _phan_tich_video(thu_muc, du_lieu, ten, conf, chu_ky, khung_moi_giay, thanh_tien_do):
    vao, tam, ra = thu_muc / f"vao{Path(ten).suffix or '.mp4'}", thu_muc / "tam.mp4", thu_muc / "ra.mp4"
    vao.write_bytes(du_lieu)

    cap = cv2.VideoCapture(str(vao))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps > 120:
        fps = 30.0
    tong = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0)  # chỉ để hiện tiến độ; vẫn đọc đến khung cuối cùng
    buoc = max(1, round(fps / khung_moi_giay))

    writer, report = None, None
    frames, vp_frames, so_lan_chay, last_alert = 0, 0, 0, float("-inf")
    canh_bao, rows = [], []
    dien_bien, dem_thieu = [], collections.Counter()  # cho biểu đồ: (thời điểm, công nhân, người thiếu) mỗi lần chạy
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = gioi_han_kich_thuoc(frame)
            thoi_diem = frames / fps
            if frames % buoc == 0:
                report, annotated = phan_tich(frame, conf)
                so_lan_chay += 1
                dien_bien.append((round(thoi_diem, 2), report["so_cong_nhan"], report["so_nguoi_vi_pham"]))
                dem_thieu.update(ten for w in report["workers"] for ten, _ in w["thieu"])
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
                thanh_tien_do.progress(min(frames / tong, 1.0) * 0.95 if tong else 0.5,
                                       text=f"Đang phân tích… {thoi_diem:.1f}s / {tong / fps:.1f}s" if tong
                                       else f"Đang phân tích… {thoi_diem:.1f}s")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if frames == 0:
        return None
    thanh_tien_do.progress(0.97, text="Đang xuất video kết quả…")
    sang_h264(tam, ra)
    return {"video": ra.read_bytes(), "ten": ten, "frames": frames, "vp_frames": vp_frames, "tong": tong,
            "fps": fps, "so_lan_chay": so_lan_chay, "canh_bao": canh_bao, "rows": rows,
            "dien_bien": dien_bien, "dem_thieu": dict(dem_thieu)}


# ---------------------------------------------------------------------- camera trực tiếp
class GiamSatCamera:
    """Trạng thái giám sát camera của một phiên, dùng chung cho chế độ A (WebRTC) và B (tự chụp).
    Chế độ A gọi xu_ly_khung_webrtc từ luồng riêng của streamlit-webrtc, nên mọi truy cập đều qua self.khoa."""

    def __init__(self):
        self.khoa = threading.Lock()
        self.conf, self.chu_ky, self.tan_suat = 0.3, 3, 4
        self.report, self.jpg_ket_qua = None, None
        self.lan_chay_cuoi, self.canh_bao_cuoi, self.ms_phan_tich = 0.0, float("-inf"), 0.0
        self.so_khung, self.so_khung_vp = 0, 0
        self.so_khung_webrtc = 0  # số khung hình máy chủ nhận được qua WebRTC (để chẩn đoán kết nối)
        self.nhat_ky, self.anh_canh_bao = [], []

    def ghi_nhan(self, report, annotated, nguon, ms):
        """Cập nhật kết quả mới nhất; ghi cảnh báo tối đa một lần mỗi `chu_ky` giây (giống notebook)."""
        now = time.time()
        with self.khoa:
            self.report, self.ms_phan_tich = report, ms
            self.so_khung += 1
            self.so_khung_vp += report["canh_bao"]
            if report["canh_bao"] and now - self.canh_bao_cuoi >= self.chu_ky:
                self.canh_bao_cuoi = now
                self.nhat_ky = dong_nhat_ky(nguon, "", report, f"camera_{datetime.now():%Y%m%d_%H%M%S}.jpg") + self.nhat_ky
                jpg = cv2.imencode(".jpg", gioi_han_kich_thuoc(annotated, 480))[1].tobytes()
                self.anh_canh_bao.insert(0, (jpg, f"{datetime.now():%H:%M:%S} · {core.mo_ta_vi_pham(report)}"))
                del self.anh_canh_bao[8:]

    def phan_tich_khung(self, bgr, nguon):
        t0 = time.perf_counter()
        report, annotated = phan_tich(bgr, self.conf)
        self.ghi_nhan(report, annotated, nguon, (time.perf_counter() - t0) * 1000)
        return report, annotated

    def xu_ly_khung_webrtc(self, frame):
        """Chế độ A: chỉ chạy mô hình `tan_suat` lần mỗi giây, các khung ở giữa vẽ lại kết quả gần nhất."""
        self.so_khung_webrtc += 1
        bgr = gioi_han_kich_thuoc(frame.to_ndarray(format="bgr24"), 640)
        now = time.monotonic()
        if self.report is None or now - self.lan_chay_cuoi >= 1 / self.tan_suat:
            self.lan_chay_cuoi = now
            _, annotated = self.phan_tich_khung(bgr, "camera (trực tiếp)")
        else:
            with self.khoa:
                report = self.report
            annotated = core.ve_ket_qua(bgr, report)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    def xoa(self):
        with self.khoa:
            self.report, self.jpg_ket_qua = None, None
            self.so_khung, self.so_khung_vp, self.canh_bao_cuoi = 0, 0, float("-inf")
            self.nhat_ky, self.anh_canh_bao = [], []


def hien_thi_trang_thai_camera(gs):
    with gs.khoa:
        report, ms, so, vp = gs.report, gs.ms_phan_tich, gs.so_khung, gs.so_khung_vp
        nhat_ky, anh_cb = list(gs.nhat_ky), list(gs.anh_canh_bao)
    if report is None:
        st.info("Đang chờ khung hình đầu tiên…", icon=":material/hourglass_top:")
        return
    the_tu_bao_cao(report)
    hien_bang_cong_nhan(report)
    st.caption(f"Đã phân tích {so} khung hình · {vp} có vi phạm ({vp / so:.0%}) · "
               f"{len(anh_cb)} lần cảnh báo gần đây · {ms:.0f} ms / lần phân tích")
    if anh_cb:
        st.image(anh_cb[0][0], caption=f"Cảnh báo gần nhất: {anh_cb[0][1]}", width="stretch")
    if nhat_ky:
        st.markdown("**:material/history: Nhật ký cảnh báo**")
        st.dataframe(pd.DataFrame(nhat_ky[:30], columns=COT_NHAT_KY), hide_index=True, width="stretch", height=220)
        st.download_button("Tải nhật ký CSV", csv_nhat_ky(nhat_ky), "camera_nhat_ky.csv", "text/csv",
                           key="csv_camera", icon=":material/download:", width="stretch")


@st.fragment(run_every=1.0)
def bang_trang_thai_truc_tiep(gs, bat_dau):
    """Chế độ A: luồng WebRTC chạy ngầm, bảng trạng thái tự làm mới mỗi giây.
    Nếu đã bấm Bắt đầu một lúc mà máy chủ chưa nhận được khung hình nào thì kết nối WebRTC không thông."""
    cho = time.time() - bat_dau
    if gs.so_khung_webrtc == 0 and cho > 8:
        st.warning(f"**Đã {cho:.0f} giây mà máy chủ chưa nhận được hình từ camera.** "
                   "Kết nối WebRTC không thông được. Thường do chưa cấu hình máy chủ chuyển tiếp TURN "
                   f"(hiện đang dùng: {nguon_ice()}) hoặc mạng chặn WebRTC. "
                   "Xem cách thêm TURN miễn phí ở README, hoặc dùng chế độ tự chụp.", icon=":material/wifi_off:")
        if st.button("Chuyển sang chế độ tự chụp mỗi giây", key="chuyen_b_canh_bao", type="primary",
                     icon=":material/photo_camera:", on_click=chuyen_sang_tu_chup):
            st.rerun()  # nút nằm trong fragment: chạy lại cả trang để đổi chế độ
        return
    hien_thi_trang_thai_camera(gs)


def bi_mat(khoa):
    try:
        return st.secrets.get(khoa)
    except Exception:  # chưa có file secrets
        return None


def may_chu_ice():
    """Danh sách STUN/TURN khai báo tay trong Secrets (khóa ice_servers), hoặc None để streamlit-webrtc tự lấy:
    TURN của Cloudflare / Twilio / Hugging Face nếu có biến môi trường tương ứng (Streamlit Cloud đưa các khóa
    cấp gốc của Secrets thành biến môi trường), nếu không thì chỉ STUN của Google."""
    tu_secrets = bi_mat("ice_servers")
    return [dict(s) for s in tu_secrets] if tu_secrets else None


def nguon_ice():
    if bi_mat("ice_servers"):
        return "máy chủ TURN tự khai báo"
    if os.getenv("CLOUDFLARE_TURN_KEY_ID") and os.getenv("CLOUDFLARE_TURN_KEY_API_TOKEN"):
        return "TURN của Cloudflare"
    if os.getenv("TWILIO_ACCOUNT_SID") and os.getenv("TWILIO_AUTH_TOKEN"):
        return "TURN của Twilio"
    if os.getenv("HF_TOKEN"):
        return "TURN của Hugging Face"
    return "chỉ STUN của Google, chưa có TURN"


CAMERA_TU_CHUP_HTML = """
<div class="khung"><video autoplay playsinline muted></video></div>
<div class="trang-thai">Đang xin quyền dùng camera…</div>
"""
CAMERA_TU_CHUP_CSS = """
.khung {border-radius: 10px; overflow: hidden; background: #0b1220; line-height: 0}
video {width: 100%; max-height: 360px; object-fit: contain}
.trang-thai {font-size: .85rem; opacity: .75; margin-top: 6px}
"""
CAMERA_TU_CHUP_JS = """
export default function(component) {
    const { data, setTriggerValue, parentElement } = component;
    const video = parentElement.querySelector('video');
    const trangThai = parentElement.querySelector('.trang-thai');
    const canvas = document.createElement('canvas');
    const chuKy = (data && data.chu_ky_ms) || 1000;
    let stream = null, timer = null, daDung = false;

    navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 } }, audio: false })
        .then((s) => {
            if (daDung) { s.getTracks().forEach((t) => t.stop()); return; }
            stream = s;
            video.srcObject = s;
            trangThai.textContent = `● Đang tự chụp và gửi lên máy chủ mỗi ${chuKy / 1000} giây`;
            timer = setInterval(() => {
                if (!video.videoWidth || document.hidden) return;
                const tiLe = Math.min(1, 640 / video.videoWidth);
                canvas.width = Math.round(video.videoWidth * tiLe);
                canvas.height = Math.round(video.videoHeight * tiLe);
                canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
                setTriggerValue('frame', canvas.toDataURL('image/jpeg', 0.8));
            }, chuKy);
        })
        .catch((err) => { trangThai.textContent = 'Không mở được camera: ' + err.message; });

    return () => {
        daDung = true;
        clearInterval(timer);
        if (stream) stream.getTracks().forEach((t) => t.stop());
    };
}
"""


@st.cache_resource
def tao_camera_tu_chup():
    return st.components.v2.component("camera_tu_chup", html=CAMERA_TU_CHUP_HTML, css=CAMERA_TU_CHUP_CSS,
                                      js=CAMERA_TU_CHUP_JS)


@st.fragment
def che_do_tu_chup(gs):
    """Chế độ B: trình duyệt tự chụp webcam mỗi giây và gửi lên qua kết nối web thường (không cần WebRTC).
    Đặt trong fragment để mỗi ảnh gửi lên chỉ chạy lại phần này, không chạy lại cả trang."""
    trai, phai = st.columns([1, 1], gap="medium")
    with trai:
        kq = tao_camera_tu_chup()(key="camera_tu_chup", data={"chu_ky_ms": 1000}, on_frame_change=lambda: None)
        anh = getattr(kq, "frame", None)
        if anh:
            bgr = cv2.imdecode(np.frombuffer(base64.b64decode(anh.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)
            if bgr is not None:
                _, annotated = gs.phan_tich_khung(bgr, "camera (tự chụp)")
                gs.jpg_ket_qua = cv2.imencode(".jpg", annotated)[1].tobytes()
        if gs.jpg_ket_qua:
            st.image(gs.jpg_ket_qua, caption="Kết quả phân tích gần nhất", width="stretch")
    with phai:
        hien_thi_trang_thai_camera(gs)


DICH_WEBRTC = {
    "start": "Bắt đầu giám sát", "stop": "Dừng", "select_device": "Chọn camera",
    "media_api_not_available": "Trình duyệt không hỗ trợ camera", "device_ask_permission": "Hãy cho phép trang dùng camera",
    "device_not_available": "Không tìm thấy camera", "device_access_denied": "Quyền dùng camera đã bị chặn",
    "turn_camera_on": "Bật camera", "turn_camera_off": "Tắt camera", "mute_microphone": "Tắt micro",
    "unmute_microphone": "Bật micro", "select_camera": "Chọn camera", "select_microphone": "Chọn micro",
}


# ---------------------------------------------------------------------- giao diện chung
CSS = """
<style>
  .block-container {padding-top: 4.5rem; max-width: 1320px}
  [data-testid="stImageCaption"] {font-size: .85rem}
  /* Phần đầu trang: khung tối cố định ở cả hai giao diện, sọc cảnh báo cam/đen như băng rào công trường */
  .hero {position: relative; overflow: hidden; display: flex; gap: 32px; justify-content: space-between;
         padding: 30px 32px 26px; margin-bottom: 1.4rem; border-radius: 18px; color: #f8fafc;
         border: 1px solid rgba(148, 163, 184, .18);
         background: radial-gradient(640px 260px at 100% 0%, rgba(234, 88, 12, .32), transparent 70%),
                     linear-gradient(135deg, #0b1220 0%, #111c33 55%, #1b2740 100%)}
  .hero::before {content: ""; position: absolute; inset: 0 0 auto 0; height: 6px;
                 background: repeating-linear-gradient(135deg, #ea580c 0 14px, #0b1220 14px 28px)}
  .hero .trai {display: flex; gap: 18px; align-items: flex-start; min-width: 0}
  .hero .logo {flex: none; display: grid; place-items: center; width: 58px; height: 58px; border-radius: 16px;
               background: linear-gradient(160deg, #f97316, #c2410c); color: #fff;
               box-shadow: 0 10px 30px -8px rgba(234, 88, 12, .6)}
  .hero .nho {font-size: .74rem; font-weight: 600; letter-spacing: .14em; text-transform: uppercase; color: #fdba74}
  .hero h1 {color: #fff; font-size: 1.9rem; line-height: 1.2; margin: .3rem 0 .45rem; padding: 0; text-wrap: balance}
  .hero p {color: #cbd5e1; max-width: 600px; margin: 0 0 .9rem; font-size: .98rem}
  .hero .the-ppe {display: inline-flex; align-items: center; gap: 6px; margin: 0 6px 6px 0; padding: 3px 11px;
                  border-radius: 999px; border: 1px solid rgba(148, 163, 184, .35); color: #e2e8f0;
                  background: rgba(15, 23, 42, .45); font-size: .8rem; font-weight: 500}
  .hero .the-ppe::before {content: ""; width: 6px; height: 6px; border-radius: 50%; background: #f97316}
  .hero .so {flex: none; display: grid; grid-template-columns: repeat(2, 148px); gap: 12px; align-self: center}
  .hero .o {padding: 12px 16px; border-radius: 12px; background: rgba(255, 255, 255, .04);
            border: 1px solid rgba(148, 163, 184, .18)}
  .hero .gia-tri {font-size: 1.55rem; font-weight: 700; color: #fff; line-height: 1.2; font-variant-numeric: tabular-nums}
  .hero .nhan {font-size: .78rem; color: #94a3b8; margin-top: 2px}
  /* Tiêu đề mục: dòng chữ nhỏ viết hoa + tiêu đề + mô tả */
  .muc {margin: .6rem 0 .5rem}
  .muc .nho {font-size: .72rem; font-weight: 600; letter-spacing: .14em; text-transform: uppercase; opacity: .6}
  .muc h3 {font-size: 1.25rem; margin: .15rem 0 .1rem; padding: 0}
  .muc p {margin: 0; opacity: .72; font-size: .92rem}
  .nhan-xet {padding: 12px 16px; border-radius: 12px; border-left: 4px solid #ea580c;
             background: rgba(234, 88, 12, .08); font-size: .93rem; margin: .4rem 0 .2rem}
  .trang-thai {display: flex; align-items: center; gap: 14px; color: #fff; border-radius: 12px;
               padding: 12px 16px; margin-bottom: .25rem}
  .trang-thai svg {flex: none}
  .trang-thai .tieu-de {font-size: 1.15rem; font-weight: 700; letter-spacing: .4px}
  .trang-thai .mo-ta {font-size: .95rem; opacity: .95}
  .chu-giai {list-style: none; padding: 0; margin: 0}
  .chu-giai li {display: flex; align-items: center; gap: 10px; margin: 0 0 .45rem; font-size: .9rem}
  .chu-giai .o {flex: none; width: 22px; height: 16px; border-radius: 3px; border: 3px solid}
  .chu-giai .o.mong {border-width: 1.5px}
  .chu-giai .ky-hieu {flex: none; width: 22px; font-size: .8rem; letter-spacing: -1px}
  .chan-trang {margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid rgba(128, 128, 128, .25);
               font-size: .82rem; opacity: .65}
  @media (max-width: 900px) {
    .hero {flex-direction: column; padding: 26px 20px 20px; gap: 18px}
    .hero .so {grid-template-columns: repeat(2, minmax(0, 1fr))}
    .hero h1 {font-size: 1.5rem}
    .hero .logo {width: 46px; height: 46px; border-radius: 13px}
  }
  @media (prefers-reduced-motion: reduce) {* {transition: none !important; animation: none !important}}
</style>
"""


def so_vn(x):
    return f"{x:,}".replace(",", ".")


def hero(nho, tieu_de, mo_ta, so_lieu):
    """Phần đầu trang: logo, tiêu đề, 4 nhãn trang bị và 4 ô số liệu nổi bật."""
    the = "".join(f'<span class="the-ppe">{ten}</span>' for ten, _, _, _ in core.PPE_RULES)
    o = "".join(f'<div class="o"><div class="gia-tri">{gt}</div><div class="nhan">{nhan}</div></div>'
                for gt, nhan in so_lieu)
    st.markdown(f'<div class="hero"><div class="trai"><div class="logo">{svg(ICON_MU, 30)}</div><div>'
                f'<div class="nho">{nho}</div><h1>{tieu_de}</h1><p>{mo_ta}</p>{the}</div></div>'
                f'<div class="so">{o}</div></div>', unsafe_allow_html=True)


def muc(nho, tieu_de, mo_ta=""):
    st.markdown(f'<div class="muc"><div class="nho">{nho}</div><h3>{tieu_de}</h3>'
                + (f"<p>{mo_ta}</p>" if mo_ta else "") + "</div>", unsafe_allow_html=True)


def nhan_xet(html):
    st.markdown(f'<div class="nhan-xet">{html}</div>', unsafe_allow_html=True)


def bieu_do(chart):
    st.altair_chart(chart, width="stretch")


def chan_trang():
    st.markdown('<div class="chan-trang">Đồ án môn Học máy · Nhóm 1 · YOLOv11n tinh chỉnh trên bộ dữ liệu PPE '
                '(Roboflow, CC BY 4.0). Ảnh và video chỉ được xử lý tạm thời, không lưu lại.</div>',
                unsafe_allow_html=True)


def tong_hop_anh(anh, conf):
    """Khi phân tích nhiều ảnh: số liệu tổng và biểu đồ trang bị bị thiếu trên toàn bộ ảnh."""
    bao_cao = [r for r in (phan_tich_anh(d, conf)[0] for _, d in anh) if r is not None]
    cong_nhan = sum(r["so_cong_nhan"] for r in bao_cao)
    vi_pham = sum(r["so_nguoi_vi_pham"] for r in bao_cao)
    dem = collections.Counter(ten for r in bao_cao for w in r["workers"] for ten, _ in w["thieu"])
    with st.container(border=True):
        muc("Tổng hợp", f"Kết quả trên {len(bao_cao)} ảnh")
        chi_so(("Ảnh có vi phạm", f"{sum(r['canh_bao'] for r in bao_cao)}/{len(bao_cao)}"),
               ("Công nhân", cong_nhan), ("Thiếu đồ bảo hộ", vi_pham),
               ("Tỷ lệ tuân thủ", bd.phan_tram((cong_nhan - vi_pham) / cong_nhan, 0) if cong_nhan else "–"))
        chart = bd.bd_trang_bi_thieu(dem, "Số công nhân thiếu")
        if chart is not None:
            st.markdown("**Trang bị bị thiếu nhiều nhất**")
            bieu_do(chart)


# ---------------------------------------------------------------------- trang 1: giám sát
NGUON_ANH = {"tai_len": ":material/upload: Tải ảnh lên", "mau": ":material/photo_library: Ảnh mẫu"}
CHE_DO_CAMERA = {"truc_tiep": ":material/sensors: Video trực tiếp", "tu_chup": ":material/photo_camera: Tự chụp mỗi giây"}


def chuyen_sang_tu_chup():
    st.session_state["che_do_camera"] = "tu_chup"


def trang_giam_sat():
    val = bd.DANH_GIA["val"]
    hero("An toàn lao động · Thị giác máy tính", "Giám sát an toàn công trường",
         "Mô hình YOLOv11n kiểm tra từng công nhân có đủ 4 trang bị bảo hộ bắt buộc và cảnh báo ngay khi thiếu, "
         "trên ảnh, video và camera trực tiếp.",
         [(bd.phan_tram(val["all"][2]), "mAP50 trên tập kiểm định"), ("9", "lớp đối tượng nhận diện"),
          (f"{str(val['toc_do_ms']).replace('.', ',')} ms", "suy luận mỗi ảnh (GPU)"),
          (so_vn(sum(bd.SO_ANH.values())), "ảnh huấn luyện và đánh giá")])

    with st.sidebar:
        st.subheader(":material/tune: Cài đặt nhận diện")
        conf = st.slider("Ngưỡng tin cậy", 0.10, 0.90, float(core.CONF_THRESHOLD), 0.05,
                         help="Thấp: phát hiện nhiều hơn nhưng dễ nhầm. Cao: chắc chắn hơn nhưng dễ bỏ sót.")
        chu_ky = st.slider("Chu kỳ cảnh báo (giây)", 1, 10, 3,
                           help="Khoảng cách tối thiểu giữa hai lần ghi cảnh báo trong video và camera.")
        khung_moi_giay = st.slider("Số lần phân tích mỗi giây", 1, 10, 4,
                                   help="Áp dụng cho video và camera. Càng cao càng chính xác nhưng càng chậm "
                                        "(máy chủ miễn phí chỉ có CPU).")
        st.divider()
        st.subheader(":material/info: Cách đọc kết quả")
        # Màu khung giống màu ve_ket_qua vẽ lên ảnh (GREEN, RED, BLUE của notebook, đổi từ BGR sang RGB)
        st.markdown("""
<ul class="chu-giai">
  <li><span class="o" style="border-color:#00aa00"></span>Công nhân đủ 4 trang bị</li>
  <li><span class="o" style="border-color:#ff0000"></span>Công nhân thiếu trang bị (ghi rõ món thiếu)</li>
  <li><span class="o mong" style="border-color:#00aaff"></span>Trang bị phát hiện được</li>
  <li><span class="o mong" style="border-color:#ff0000"></span>Dấu hiệu vi phạm (không mũ, tay trần…)</li>
  <li><b class="ky-hieu">✔ ✘</b>Bảng công nhân: có / thiếu trang bị</li>
</ul>
""", unsafe_allow_html=True)
        st.caption("Không có lớp *no-gloves* nên thiếu găng tay được suy ra từ việc không phát hiện găng tay.")

    tab_anh, tab_video, tab_camera = st.tabs([":material/image: Ảnh", ":material/movie: Video",
                                              ":material/videocam: Camera"])
    with tab_anh:
        nguon = st.segmented_control("Nguồn ảnh", list(NGUON_ANH), format_func=NGUON_ANH.get, default="tai_len",
                                     required=True, label_visibility="collapsed")
        if nguon == "mau":
            mau = sorted((APP_DIR / "vi_du").glob("*.jpg"))
            nhan = {p.name: f"Mẫu {i}" for i, p in enumerate(mau, 1)}
            chon = st.pills("Chọn ảnh mẫu", list(nhan), format_func=nhan.get, default=mau[0].name if mau else None,
                            required=True)
            anh = [(chon, (APP_DIR / "vi_du" / chon).read_bytes())] if chon else []
        else:
            tep = st.file_uploader("Kéo thả hoặc chọn một / nhiều ảnh công trường",
                                   type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=True)
            anh = [(f.name, f.getvalue()) for f in tep or []]
        if not anh:
            st.info("Chọn ảnh để bắt đầu (có thể chọn nhiều ảnh cùng lúc), hoặc chuyển sang **Ảnh mẫu** để xem thử.",
                    icon=":material/add_photo_alternate:")
        if len(anh) > 1:
            with st.spinner(f"Đang phân tích {len(anh)} ảnh…"):
                tong_hop_anh(anh, conf)
        for i, (ten, du_lieu) in enumerate(anh):
            with st.container(border=True):
                with st.spinner(f"Đang phân tích {ten}…"):
                    hien_thi_ket_qua_anh(ten, du_lieu, conf, f"a{i}")

    with tab_video:
        tab_video_noi_dung(conf, chu_ky, khung_moi_giay)

    with tab_camera:
        tab_camera_noi_dung(conf, chu_ky, khung_moi_giay)
    chan_trang()


def tab_video_noi_dung(conf, chu_ky, khung_moi_giay):
    tep_video = st.file_uploader("Chọn video công trường từ máy tính", type=["mp4", "avi", "mov", "mkv"])
    if tep_video is None:
        st.info("Tải video lên để bắt đầu. Kết quả gồm video đã đánh dấu, biểu đồ diễn biến vi phạm, ảnh các "
                "khoảnh khắc vi phạm và nhật ký cảnh báo.", icon=":material/video_file:")
        return
    duoi = Path(tep_video.name).suffix.lower()
    giay = thoi_luong_video(tep_video.file_id, duoi, tep_video.getvalue())
    if giay:
        # máy chủ miễn phí chỉ có CPU: khoảng 2–6 lần độ dài video khi phân tích 4 lần mỗi giây
        he_so = khung_moi_giay / 4
        it, nhieu = max(1, round(giay * 2 * he_so / 60)), max(1, round(giay * 6 * he_so / 60))
        st.caption(f"Video dài **{dinh_dang_thoi_luong(giay)}**, được phân tích toàn bộ nên video kết quả dài "
                   f"đúng bằng video gốc. Máy chủ miễn phí chỉ có CPU, ước tính mất khoảng **{it}–{nhieu} phút**. "
                   "Đừng đổi cài đặt trong lúc phân tích vì trang sẽ chạy lại và dừng phân tích.")
    if st.button("Phân tích video", type="primary", icon=":material/play_arrow:"):
        thanh = st.progress(0.0, text="Đang chuẩn bị…")
        kq = phan_tich_video(tep_video.getvalue(), tep_video.name, conf, chu_ky, khung_moi_giay, thanh)
        thanh.empty()
        if kq is None:
            st.error("Không đọc được video. Hãy thử file .mp4 khác.", icon=":material/error:")
        else:
            kq["file_id"] = tep_video.file_id
        st.session_state["video_kq"] = kq

    kq = st.session_state.get("video_kq")
    if not kq or kq.get("file_id") != tep_video.file_id:
        kq = None  # kết quả cũ thuộc video khác
    goc, ket_qua = st.columns(2, gap="medium")
    with goc:
        st.markdown("**:material/movie: Video gốc**")
        with st.container(key="video_goc"):
            st.video(ban_xem_truoc(tep_video.file_id, duoi, tep_video.getvalue()))
    with ket_qua:
        st.markdown("**:material/verified_user: Video đã phân tích**")
        if kq:
            with st.container(key="video_ket_qua"):
                st.video(kq["video"], muted=True)  # không có tiếng; tiếng phát từ video gốc
        else:
            st.info("Bấm **Phân tích video** để xem video có đánh dấu công nhân và cảnh báo.",
                    icon=":material/play_circle:")
    if not kq:
        return

    tao_dong_bo_video()(key="dong_bo_video", data={"lop": ["st-key-video_goc", "st-key-video_ket_qua"]})
    st.caption("Hai video phát cùng lúc: bấm phát, tạm dừng hoặc tua ở một video thì video kia làm theo.")
    with st.container(border=True):
        mo_ta = f"{kq['vp_frames']}/{kq['frames']} khung hình có công nhân thiếu đồ bảo hộ"
        if kq["vp_frames"]:
            the_trang_thai("do", "CÓ VI PHẠM", mo_ta)
        else:
            the_trang_thai("xanh", "AN TOÀN", "Không khung hình nào có vi phạm")
        chi_so(("Đã phân tích", dinh_dang_thoi_luong(kq["frames"] / kq["fps"])),
               ("Khung có vi phạm", bd.phan_tram(kq["vp_frames"] / kq["frames"], 0)),
               ("Lần cảnh báo", len({r[2] for r in kq["rows"]})),
               ("Lần chạy mô hình", kq["so_lan_chay"]))
        if kq["tong"] and kq["frames"] < 0.98 * kq["tong"]:  # file hỏng giữa chừng
            st.caption(f"Chỉ đọc được {dinh_dang_thoi_luong(kq['frames'] / kq['fps'])} trong tổng "
                       f"{dinh_dang_thoi_luong(kq['tong'] / kq['fps'])} của video.")
        ten_goc = Path(kq["ten"]).stem
        nut1, nut2 = st.columns(2)
        nut1.download_button("Tải video kết quả", kq["video"], f"{ten_goc}_giam_sat.mp4", "video/mp4",
                             icon=":material/download:", width="stretch")
        nut2.download_button("Tải nhật ký CSV", csv_nhat_ky(kq["rows"]), f"{ten_goc}_nhat_ky.csv",
                             "text/csv", disabled=not kq["rows"], icon=":material/table_view:", width="stretch")

    if kq.get("dien_bien"):
        muc("Phân tích", "Diễn biến trong video",
            "Gộp theo từng giây, lấy số lớn nhất trong giây đó. Rê chuột lên biểu đồ để xem số liệu từng giây.")
        trai, phai = st.columns([3, 2], gap="medium")
        with trai, st.container(border=True):
            st.markdown("**Công nhân và người thiếu đồ bảo hộ theo từng giây**")
            bieu_do(bd.bd_dien_bien(kq["dien_bien"]))
        with phai, st.container(border=True):
            st.markdown("**Trang bị bị thiếu nhiều nhất**")
            chart = bd.bd_trang_bi_thieu(kq.get("dem_thieu", {}), "Lượt công nhân thiếu")
            if chart is None:
                st.success("Không có công nhân nào thiếu trang bị.", icon=":material/verified:")
            else:
                bieu_do(chart)
                st.caption("Đếm theo lượt: một công nhân thiếu trong một lần chạy mô hình tính là một lượt.")
    if kq["canh_bao"]:
        muc("Bằng chứng", "Khoảnh khắc vi phạm", "Ảnh chụp tại mỗi lần cảnh báo (tối đa 12 ảnh).")
        cot = st.columns(4)
        for i, (jpg, chu_thich) in enumerate(kq["canh_bao"]):
            cot[i % 4].image(jpg, caption=chu_thich, width="stretch")
        muc("Nhật ký", "Nhật ký cảnh báo")
        st.dataframe(pd.DataFrame(kq["rows"], columns=COT_NHAT_KY), hide_index=True, width="stretch")


def tab_camera_noi_dung(conf, chu_ky, khung_moi_giay):
    gs = st.session_state.setdefault("giam_sat_camera", GiamSatCamera())
    gs.conf, gs.chu_ky, gs.tan_suat = conf, chu_ky, khung_moi_giay

    st.session_state.setdefault("che_do_camera", "truc_tiep")  # mặc định: video trực tiếp (A)
    tren_trai, tren_phai = st.columns([3, 2], vertical_alignment="center")
    with tren_trai:
        che_do = st.segmented_control("Chế độ camera", list(CHE_DO_CAMERA), format_func=CHE_DO_CAMERA.get,
                                      key="che_do_camera", required=True, label_visibility="collapsed")
    with tren_phai:
        st.button("Xóa nhật ký camera", icon=":material/delete_sweep:", on_click=gs.xoa, width="stretch")

    if che_do == "truc_tiep":
        st.caption("Bấm **Bắt đầu giám sát** rồi cho phép dùng camera. Video được gửi lên máy chủ qua WebRTC, "
                   f"mô hình chạy khoảng {khung_moi_giay} lần/giây (chỉnh ở thanh bên trái).")
        trai, phai = st.columns([1, 1], gap="medium")
        ice = may_chu_ice()
        with trai:
            ctx = webrtc_streamer(
                key="camera_truc_tiep", mode=WebRtcMode.SENDRECV,
                rtc_configuration={"iceServers": ice} if ice else None,
                video_frame_callback=gs.xu_ly_khung_webrtc,
                media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480},
                                                    "frameRate": {"ideal": 15, "max": 20}}, "audio": False},
                async_processing=True, translations=DICH_WEBRTC,
                video_html_attrs={"style": {"width": "100%", "borderRadius": "10px"}, "autoPlay": True,
                                  "controls": False, "muted": True})
            st.button("Không lên hình? Chuyển sang chế độ tự chụp mỗi giây", icon=":material/sync_alt:",
                      on_click=chuyen_sang_tu_chup, width="stretch")
            st.caption(f"Máy chủ kết nối: **{nguon_ice()}**. Nếu bấm Bắt đầu mà không lên hình, cần thêm máy chủ "
                       "TURN (xem README) hoặc dùng chế độ tự chụp, chạy được trên mọi mạng.")
        with phai:
            if ctx.state.playing:
                if "webrtc_bat_dau" not in st.session_state:  # vừa bấm Bắt đầu
                    st.session_state["webrtc_bat_dau"] = time.time()
                    gs.so_khung_webrtc = 0
                bang_trang_thai_truc_tiep(gs, st.session_state["webrtc_bat_dau"])
            else:
                st.session_state.pop("webrtc_bat_dau", None)
                hien_thi_trang_thai_camera(gs)
    else:
        st.caption("Trình duyệt tự chụp webcam **mỗi giây** và gửi lên phân tích qua kết nối web thường, "
                   "chạy được cả khi mạng chặn WebRTC. Kết quả cập nhật khoảng 1 lần/giây.")
        if st.toggle("Bật camera", key="bat_tu_chup"):
            che_do_tu_chup(gs)
        elif gs.report is not None:
            hien_thi_trang_thai_camera(gs)
        else:
            st.info("Bật công tắc **Bật camera** để bắt đầu.", icon=":material/toggle_on:")

    st.info("**Cần hình mượt nhất khi demo trước lớp?** Dùng app desktop trên laptop có GPU: "
            "`.venv\\Scripts\\python.exe giao_dien.py` → chế độ **Camera** (khoảng 28 khung hình/giây, "
            "có tạm dừng, âm thanh cảnh báo và lưu nhật ký).", icon=":material/lightbulb:")


# ---------------------------------------------------------------------- trang 2: hiệu năng mô hình
ANH_ULTRALYTICS = {
    "confusion_matrix_normalized.png": ("Ma trận nhầm lẫn",
                                        "Tỷ lệ dự đoán (cột) cho mỗi lớp thật (hàng), đã chuẩn hóa. Đường chéo "
                                        "càng đậm càng tốt; ô ngoài đường chéo là nhầm lẫn giữa hai lớp."),
    "BoxPR_curve.png": ("Đường Precision–Recall",
                        "Diện tích dưới mỗi đường là AP của lớp đó; mAP50 là trung bình của 9 lớp."),
    "BoxF1_curve.png": ("Đường F1 theo ngưỡng tin cậy",
                        "Đỉnh của đường tổng cho biết ngưỡng tin cậy cân bằng Precision và Recall tốt nhất."),
    "BoxP_curve.png": ("Precision theo ngưỡng", "Precision của từng lớp khi tăng ngưỡng tin cậy."),
    "BoxR_curve.png": ("Recall theo ngưỡng", "Recall của từng lớp khi tăng ngưỡng tin cậy."),
}


def trang_mo_hinh():
    df = bd.doc_ket_qua_huan_luyen()
    test = bd.DANH_GIA["test"]
    hero("Đánh giá mô hình · YOLOv11n", "Hiệu năng mô hình",
         "Kết quả huấn luyện 50 epoch và đánh giá trên hai tập dữ liệu chưa dùng để huấn luyện: "
         "tập kiểm định (val) và tập kiểm tra (test).",
         [(bd.phan_tram(test["all"][2]), "mAP50 trên tập kiểm tra"),
          (bd.phan_tram(test["all"][3]), "mAP50-95 trên tập kiểm tra"),
          (f"{round(df['time'].iloc[-1] / 60)} phút", "thời gian huấn luyện"),
          (so_vn(bd.DANH_GIA["val"]["doi_tuong"] + test["doi_tuong"]), "đối tượng dùng để đánh giá")])

    tap = st.segmented_control("Tập đánh giá", list(bd.TEN_TAP), format_func=bd.TEN_TAP.get, default="test",
                               required=True, key="tap_danh_gia")
    khac = "val" if tap == "test" else "test"
    so, so_khac = bd.DANH_GIA[tap], bd.DANH_GIA[khac]
    with st.container(horizontal=True, gap="small"):
        for i, ten in enumerate(bd.CHI_SO):
            chenh = (so["all"][i] - so_khac["all"][i]) * 100
            # mũi tên của st.metric đã cho biết cao hơn hay thấp hơn, nên chỉ ghi độ lớn
            st.metric(ten, bd.phan_tram(so["all"][i]), delta=f"{abs(chenh):.1f} điểm so với {khac}".replace(".", ",")
                      if chenh >= 0 else f"-{abs(chenh):.1f} điểm so với {khac}".replace(".", ","),
                      delta_color="off", border=True, width="stretch")
    st.caption(f"{so_vn(so['anh'])} ảnh, {so_vn(so['doi_tuong'])} đối tượng gắn nhãn · tốc độ suy luận "
               f"{str(so['toc_do_ms']).replace('.', ',')} ms/ảnh trên RTX 3050.")

    # --- theo từng lớp
    muc("Theo từng lớp", "Hiệu năng của 9 lớp đối tượng",
        "So sánh tập kiểm định và tập kiểm tra. Lớp có đánh dấu ✔ trong bảng là trang bị bắt buộc.")
    chi_so_chon = st.pills("Chỉ số", list(bd.CHI_SO), default="mAP50", required=True, key="chi_so_lop")
    i = bd.CHI_SO.index(chi_so_chon)
    xep = sorted(bd.DANH_GIA[tap]["lop"].items(), key=lambda kv: kv[1][i])
    yeu = [f"<b>{bd.TEN_LOP[lop]}</b> ({bd.phan_tram(v[i])})" for lop, v in xep[:2]]
    manh = xep[-1]
    nhan_xet(f"Trên tập {bd.TEN_TAP[tap].lower()}, lớp tốt nhất theo {chi_so_chon} là <b>{bd.TEN_LOP[manh[0]]}</b> "
             f"({bd.phan_tram(manh[1][i])}); yếu nhất là {yeu[0]} và {yeu[1]}. Găng tay và giày là vật nhỏ, hay bị "
             "che khuất nên khó nhận diện hơn; lớp <i>no-boot</i> lại có rất ít mẫu huấn luyện (461 đối tượng).")
    with st.container(border=True):
        bieu_do(bd.bd_theo_lop(chi_so_chon))
    with st.expander("Xem bảng số liệu theo lớp", icon=":material/table_view:"):
        st.dataframe(bd.bang_danh_gia(tap).style.format({c: "{:.3f}" for c in bd.CHI_SO}), hide_index=True,
                     width="stretch")

    trai, phai = st.columns(2, gap="medium")
    with trai:
        muc("Tổng quan", "Bản đồ nhiệt lớp × chỉ số", f"Tập {bd.TEN_TAP[tap].lower()}, xếp theo mAP50.")
        with st.container(border=True):
            bieu_do(bd.bd_nhiet(tap))
    with phai:
        muc("Dữ liệu", "Phân bố đối tượng gắn nhãn",
            f"{so_vn(sum(bd.SO_ANH.values()))} ảnh từ Roboflow, chia thành ba tập.")
        with st.container(border=True):
            bieu_do(bd.bd_phan_bo())
        st.caption("Lớp *no-boot* chỉ có 651 đối tượng, ít hơn khoảng 25 lần so với *Person*: dữ liệu mất cân bằng.")

    # --- quá trình huấn luyện
    muc("Huấn luyện", "Quá trình huấn luyện qua 50 epoch",
        "Chỉ số đo trên tập kiểm định sau mỗi epoch. Rê chuột lên biểu đồ để xem giá trị của từng epoch.")
    trai, phai = st.columns(2, gap="medium")
    with trai, st.container(border=True):
        st.markdown("**Precision, Recall và mAP**")
        bieu_do(bd.bd_huan_luyen(df))
    with phai, st.container(border=True):
        loai = st.segmented_control("Hàm mất mát", ["box", "cls", "dfl"], default="box", required=True,
                                    key="loai_loss",
                                    format_func={"box": "Box (vị trí)", "cls": "Cls (phân lớp)",
                                                 "dfl": "DFL (biên hộp)"}.get)
        bieu_do(bd.bd_mat_mat(df, loai))
    tot_nhat = df.loc[df["metrics/mAP50(B)"].idxmax()]
    with st.container(horizontal=True, gap="small"):
        for nhan, gia_tri in [*bd.CAU_HINH_HUAN_LUYEN.items(),
                              (f"mAP50 cao nhất (epoch {int(tot_nhat['epoch'])})",
                               bd.phan_tram(tot_nhat["metrics/mAP50(B)"]))]:
            st.metric(nhan, gia_tri, border=True, width="stretch")

    # --- biểu đồ gốc
    muc("Ultralytics", "Biểu đồ đánh giá gốc",
        f"Do Ultralytics tạo ra khi chạy <code>model.val()</code> trên tập {bd.TEN_TAP[tap].lower()}.")
    tabs = st.tabs([ten for ten, _ in ANH_ULTRALYTICS.values()] + ["Kết quả huấn luyện"])
    for t, (tep, (_, giai_thich)) in zip(tabs, ANH_ULTRALYTICS.items()):
        with t:
            st.caption(giai_thich)
            st.image(str(bd.THU_MUC / tap / tep), width="stretch")
    with tabs[-1]:
        st.caption("Tổng hợp loss và chỉ số theo epoch do Ultralytics vẽ (results.png).")
        st.image(str(bd.THU_MUC / "results.png"), width="stretch")
    chan_trang()


st.markdown(CSS, unsafe_allow_html=True)
trang = st.navigation([
    st.Page(trang_giam_sat, title="Giám sát", icon=":material/videocam:", url_path="giam-sat", default=True),
    st.Page(trang_mo_hinh, title="Hiệu năng mô hình", icon=":material/monitoring:", url_path="hieu-nang-mo-hinh"),
], position="top")
trang.run()

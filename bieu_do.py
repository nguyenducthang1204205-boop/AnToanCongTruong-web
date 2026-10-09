"""Số liệu và biểu đồ phân tích (Altair) cho bản web.

Màu các loạt dữ liệu không viết cứng ở đây: Altair dùng chartCategoricalColors của theme trong
.streamlit/config.toml (bảng màu đã kiểm tra mù màu cho cả nền sáng và nền tối), theo thứ tự cố định của `domain`
nên mỗi loạt luôn giữ một màu dù lọc thế nào.
"""
from pathlib import Path

import altair as alt
import pandas as pd

THU_MUC = Path(__file__).resolve().parent / "assets" / "mo_hinh"

TEN_LOP = {"Gloves": "Găng tay", "Helmet": "Mũ bảo hộ", "Person": "Người", "Safety Boot": "Giày bảo hộ",
           "Safety Vest": "Áo bảo hộ", "bare-arms": "Tay trần", "no-boot": "Không giày", "no-helmet": "Không mũ",
           "no-vest": "Không áo"}
BAT_BUOC = ("Helmet", "Gloves", "Safety Vest", "Safety Boot")
CHI_SO = ("Precision", "Recall", "mAP50", "mAP50-95")
TEN_TAP = {"val": "Kiểm định (val)", "test": "Kiểm tra (test)"}

# Kết quả model.val trên tập val và test (ô đánh giá của notebook, runs/detect/eval_val và eval_test):
# (Precision, Recall, mAP50, mAP50-95)
DANH_GIA = {
    "val": {"anh": 1584, "doi_tuong": 14796, "toc_do_ms": 6.4, "all": (0.843, 0.800, 0.835, 0.467), "lop": {
        "Gloves": (0.774, 0.693, 0.735, 0.358), "Helmet": (0.932, 0.933, 0.950, 0.580),
        "Person": (0.896, 0.931, 0.928, 0.638), "Safety Boot": (0.773, 0.679, 0.742, 0.356),
        "Safety Vest": (0.848, 0.889, 0.912, 0.586), "bare-arms": (0.833, 0.752, 0.795, 0.338),
        "no-boot": (0.853, 0.763, 0.798, 0.546), "no-helmet": (0.871, 0.774, 0.828, 0.362),
        "no-vest": (0.805, 0.785, 0.826, 0.442)}},
    "test": {"anh": 858, "doi_tuong": 8248, "toc_do_ms": 6.4, "all": (0.820, 0.763, 0.807, 0.436), "lop": {
        "Gloves": (0.765, 0.672, 0.707, 0.340), "Helmet": (0.939, 0.915, 0.939, 0.547),
        "Person": (0.925, 0.912, 0.936, 0.627), "Safety Boot": (0.761, 0.678, 0.741, 0.344),
        "Safety Vest": (0.874, 0.893, 0.917, 0.565), "bare-arms": (0.794, 0.669, 0.717, 0.295),
        "no-boot": (0.749, 0.625, 0.736, 0.437), "no-helmet": (0.760, 0.676, 0.712, 0.312),
        "no-vest": (0.815, 0.831, 0.857, 0.458)}},
}
# Số đối tượng gắn nhãn của từng lớp (đếm từ dataset/*/labels) và số ảnh mỗi tập
PHAN_BO = {
    "Huấn luyện": {"Gloves": 4769, "Helmet": 10728, "Person": 11306, "Safety Boot": 7270, "Safety Vest": 7215,
                           "bare-arms": 5615, "no-boot": 461, "no-helmet": 1164, "no-vest": 3983},
    "Kiểm định": {"Gloves": 1383, "Helmet": 3036, "Person": 3184, "Safety Boot": 2007, "Safety Vest": 2042,
                        "bare-arms": 1574, "no-boot": 118, "no-helmet": 314, "no-vest": 1138},
    "Kiểm tra": {"Gloves": 680, "Helmet": 1738, "Person": 1895, "Safety Boot": 1083, "Safety Vest": 1139,
                        "bare-arms": 848, "no-boot": 72, "no-helmet": 188, "no-vest": 605},
}
SO_ANH = {"Huấn luyện": 5667, "Kiểm định": 1584, "Kiểm tra": 858}
CAU_HINH_HUAN_LUYEN = {"Mô hình gốc": "YOLOv11n (COCO)", "Số epoch": "50", "Kích thước ảnh": "640 px",
                       "Batch": "16", "Tốc độ học ban đầu": "0,01", "GPU": "RTX 3050 6 GB"}

AN_TRUC = alt.Axis(grid=False, ticks=False, domain=False, labelPadding=6, labelLimit=240)


# Biểu thức Vega cho số kiểu Việt Nam (dấu phẩy thập phân). Tên trường có dấu cách / dấu tiếng Việt nên phải
# viết datum['…'], không viết datum.… được.
def PHAN_TRAM_VN(truong):
    return f"replace(format(datum[{truong!r}], '.1%'), '.', ',')"


def SO_VN(truong):
    return f"replace(format(datum[{truong!r}], '.3f'), '.', ',')"


def phan_tram(x, so_le=1):
    return f"{x * 100:.{so_le}f}%".replace(".", ",")


def nhan_lop(lop):
    return f"{TEN_LOP[lop]} ({lop})"


def bang_danh_gia(tap):
    """Bảng chỉ số từng lớp của một tập (dùng cho bảng xem số liệu)."""
    rows = [{"Lớp": nhan_lop(lop), "Bắt buộc": "✔" if lop in BAT_BUOC else "", **dict(zip(CHI_SO, v))}
            for lop, v in DANH_GIA[tap]["lop"].items()]
    return pd.DataFrame(rows)


def doc_ket_qua_huan_luyen():
    df = pd.read_csv(THU_MUC / "results.csv")
    df.columns = [c.strip() for c in df.columns]
    return df


# ---------------------------------------------------------------------- trang hiệu năng mô hình
def bd_theo_lop(chi_so):
    """Thanh ngang ghép đôi: chỉ số `chi_so` của từng lớp trên tập val và test (một trục 0–100%)."""
    rows = [{"Lớp": TEN_LOP[lop], "Lớp gốc": lop, "Tập": TEN_TAP[tap],
             "Giá trị": DANH_GIA[tap]["lop"][lop][CHI_SO.index(chi_so)],
             "Nhóm": "Trang bị bắt buộc" if lop in BAT_BUOC else "Lớp khác"}
            for tap in ("val", "test") for lop in DANH_GIA[tap]["lop"]]
    df = pd.DataFrame(rows)
    thu_tu = (df[df["Tập"] == TEN_TAP["val"]].sort_values("Giá trị", ascending=False)["Lớp"].tolist())
    return (alt.Chart(df).mark_bar(cornerRadiusEnd=4)
            .transform_calculate(hien_thi=PHAN_TRAM_VN("Giá trị"))
            .encode(y=alt.Y("Lớp:N", sort=thu_tu, title=None, axis=AN_TRUC,
                            scale=alt.Scale(paddingInner=0.28, paddingOuter=0.1)),
                    yOffset=alt.YOffset("Tập:N", sort=list(TEN_TAP.values()), scale=alt.Scale(paddingInner=0.12)),
                    x=alt.X("Giá trị:Q", title=chi_so, scale=alt.Scale(domain=[0, 1]),
                            axis=alt.Axis(format="%", tickCount=5)),
                    color=alt.Color("Tập:N", scale=alt.Scale(domain=list(TEN_TAP.values())),
                                    legend=alt.Legend(orient="top", title=None)),
                    tooltip=[alt.Tooltip("Lớp:N"), alt.Tooltip("Lớp gốc:N"), alt.Tooltip("Nhóm:N"), alt.Tooltip("Tập:N"),
                             alt.Tooltip("hien_thi:N", title=chi_so)])
            .properties(height=440))


def bd_nhiet(tap):
    """Bản đồ nhiệt lớp × chỉ số (một sắc xanh, nhạt → đậm), có ghi giá trị trong từng ô."""
    rows = [{"Lớp": TEN_LOP[lop], "Lớp gốc": lop, "Chỉ số": cs, "Giá trị": v[i]}
            for lop, v in DANH_GIA[tap]["lop"].items() for i, cs in enumerate(CHI_SO)]
    df = pd.DataFrame(rows)
    thu_tu = [TEN_LOP[lop] for lop, _ in sorted(DANH_GIA[tap]["lop"].items(), key=lambda kv: -kv[1][2])]
    goc = alt.Chart(df).transform_calculate(hien_thi=PHAN_TRAM_VN("Giá trị")).encode(
        x=alt.X("Chỉ số:N", sort=list(CHI_SO), title=None, axis=alt.Axis(orient="top", labelAngle=0, **_khong_ke())),
        y=alt.Y("Lớp:N", sort=thu_tu, title=None, axis=AN_TRUC))
    o = goc.mark_rect(cornerRadius=3, stroke=None).encode(
        color=alt.Color("Giá trị:Q", title=None, scale=alt.Scale(domain=[0.25, 1], range=["#cde2fb", "#184f95"]),
                        legend=alt.Legend(format="%", orient="bottom", gradientLength=220)),
        tooltip=[alt.Tooltip("Lớp:N"), alt.Tooltip("Lớp gốc:N"), alt.Tooltip("Chỉ số:N"),
                 alt.Tooltip("hien_thi:N", title="Giá trị")])
    chu = goc.mark_text(fontSize=12, fontWeight=500).encode(
        text="hien_thi:N",
        color=alt.condition(alt.datum["Giá trị"] > 0.62, alt.value("#ffffff"), alt.value("#0f172a")))
    return (o + chu).properties(height=400)


def _khong_ke():
    return {"ticks": False, "domain": False, "grid": False, "labelPadding": 8}


def bd_huan_luyen(df):
    """Đường cong Precision / Recall / mAP50 / mAP50-95 trên tập val theo epoch, rê chuột để xem cả 4 giá trị."""
    cot = {"metrics/mAP50(B)": "mAP50", "metrics/mAP50-95(B)": "mAP50-95",
           "metrics/precision(B)": "Precision", "metrics/recall(B)": "Recall"}
    rong = df[["epoch", *cot]].rename(columns=cot)
    dai = rong.melt("epoch", var_name="Chỉ số", value_name="Giá trị")
    thu_tu = list(cot.values())
    return _duong_co_do_doc(dai, rong, thu_tu, "Giá trị", alt.Axis(format="%", tickCount=5),
                            alt.Scale(domain=[0, 1]), PHAN_TRAM_VN)


def bd_mat_mat(df, loai):
    """Hàm mất mát `loai` (box / cls / dfl) trên tập huấn luyện và tập val theo epoch."""
    cot = {f"train/{loai}_loss": "Huấn luyện", f"val/{loai}_loss": "Kiểm định"}
    rong = df[["epoch", *cot]].rename(columns=cot)
    dai = rong.melt("epoch", var_name="Tập", value_name="Loss")
    return _duong_co_do_doc(dai, rong, list(cot.values()), "Loss", alt.Axis(tickCount=5),
                            alt.Scale(zero=False), SO_VN, ten_mau="Tập", nhan_cuoi=True)


def _duong_co_do_doc(dai, rong, thu_tu, truong_y, truc_y, thang_y, dinh_dang, ten_mau="Chỉ số", nhan_cuoi=False):
    """Biểu đồ đường nhiều loạt + đường dóng dọc theo epoch gần con trỏ nhất, tooltip ghi đủ giá trị các loạt."""
    chon = alt.selection_point(nearest=True, on="pointerover", fields=["epoch"], empty=False, clear="pointerout")
    goc = alt.Chart(dai).encode(
        x=alt.X("epoch:Q", title="Epoch", scale=alt.Scale(domain=[1, int(dai["epoch"].max())], nice=False),
                axis=alt.Axis(tickCount=10, grid=False)),
        y=alt.Y(f"{truong_y}:Q", title=None, scale=thang_y, axis=truc_y),
        # từ 3 loạt trở lên chú thích chia 2 cột, nếu không sẽ bị cắt trên màn hình điện thoại
        color=alt.Color(f"{ten_mau}:N", scale=alt.Scale(domain=thu_tu),
                        legend=alt.Legend(orient="top", title=None, columns=2 if len(thu_tu) > 2 else 0)))
    duong = goc.mark_line(strokeWidth=2, interpolate="monotone")
    diem = goc.mark_point(filled=True, size=70, stroke="white", strokeWidth=2).encode(
        opacity=alt.condition(chon, alt.value(1), alt.value(0)))
    tooltip = [alt.Tooltip("epoch:Q", title="Epoch")] + [alt.Tooltip(f"_{i}:N", title=ten)
                                                        for i, ten in enumerate(thu_tu)]
    doc = alt.Chart(rong)
    for i, ten in enumerate(thu_tu):
        doc = doc.transform_calculate(**{f"_{i}": dinh_dang(ten)})
    doc = doc.mark_rule(strokeWidth=1, color="#94a3b8").encode(
        x="epoch:Q", opacity=alt.condition(chon, alt.value(0.8), alt.value(0)), tooltip=tooltip).add_params(chon)
    lop = duong + diem + doc
    if nhan_cuoi:  # ghi trực tiếp giá trị ở epoch cuối (chữ màu thường) khi các đường đủ xa nhau
        lop += goc.transform_filter(alt.datum.epoch == int(dai["epoch"].max())).transform_calculate(
            nhan=f"datum[{ten_mau!r}] + ' ' + " + dinh_dang(truong_y)
        ).mark_text(align="left", dx=8, fontSize=11).encode(text="nhan:N", color=alt.value("#64748b"))
    return lop.properties(height=320, padding={"right": 110 if nhan_cuoi else 8, "left": 4, "top": 4, "bottom": 4})


def bd_phan_bo():
    """Số đối tượng gắn nhãn của từng lớp, chồng theo tập train / val / test."""
    rows = [{"Lớp": TEN_LOP[lop], "Lớp gốc": lop, "Tập": tap, "Số đối tượng": so, "Thứ tự": i}
            for i, tap in enumerate(PHAN_BO) for lop, so in PHAN_BO[tap].items()]
    df = pd.DataFrame(rows)
    tong = df.groupby("Lớp")["Số đối tượng"].sum().sort_values(ascending=False).index.tolist()
    return (alt.Chart(df).mark_bar(height={"band": 0.62}, stroke=None)
            .encode(y=alt.Y("Lớp:N", sort=tong, title=None, axis=AN_TRUC),
                    x=alt.X("sum(Số đối tượng):Q", title="Số đối tượng gắn nhãn", axis=alt.Axis(format=",d", tickCount=6)),
                    color=alt.Color("Tập:N", scale=alt.Scale(domain=list(PHAN_BO)),
                                    legend=alt.Legend(orient="top", title=None, columns=2)),
                    order=alt.Order("Thứ tự:Q"),
                    tooltip=[alt.Tooltip("Lớp:N"), alt.Tooltip("Lớp gốc:N"), alt.Tooltip("Tập:N"),
                             alt.Tooltip("Số đối tượng:Q", format=",d")])
            .properties(height=380))


# ---------------------------------------------------------------------- phân tích kết quả giám sát
def bd_dien_bien(dong):
    """Số công nhân và số người thiếu đồ bảo hộ theo từng giây của video. Mô hình chạy vài lần mỗi giây; gộp theo
    giây và lấy giá trị lớn nhất (thận trọng cho an toàn) để đường không bị rối."""
    df = pd.DataFrame(dong, columns=["Thời điểm (s)", "Công nhân", "Thiếu đồ bảo hộ"])
    df["Thời điểm (s)"] = df["Thời điểm (s)"].astype(int)
    df = df.groupby("Thời điểm (s)", as_index=False).max()
    rong = df.copy()
    dai = df.melt("Thời điểm (s)", var_name="Loạt", value_name="Số người")
    thu_tu = ["Công nhân", "Thiếu đồ bảo hộ"]
    chon = alt.selection_point(nearest=True, on="pointerover", fields=["Thời điểm (s)"], empty=False,
                               clear="pointerout")
    goc = alt.Chart(dai).encode(
        x=alt.X("Thời điểm (s):Q", title="Thời điểm trong video (giây)", axis=alt.Axis(grid=False, tickCount=8)),
        y=alt.Y("Số người:Q", title=None, axis=alt.Axis(tickMinStep=1, format="d")),
        color=alt.Color("Loạt:N", scale=alt.Scale(domain=thu_tu), legend=alt.Legend(orient="top", title=None)))
    duong = goc.mark_line(strokeWidth=2, interpolate="step-after")
    diem = goc.mark_point(filled=True, size=70, stroke="white", strokeWidth=2).encode(
        opacity=alt.condition(chon, alt.value(1), alt.value(0)))
    doc = alt.Chart(rong).mark_rule(strokeWidth=1, color="#94a3b8").encode(
        x="Thời điểm (s):Q", opacity=alt.condition(chon, alt.value(0.8), alt.value(0)),
        tooltip=[alt.Tooltip("Thời điểm (s):Q", title="Giây thứ"), alt.Tooltip("Công nhân:Q"),
                 alt.Tooltip("Thiếu đồ bảo hộ:Q")]).add_params(chon)
    return (duong + diem + doc).properties(height=260)


def bd_trang_bi_thieu(dem, don_vi):
    """Thanh ngang: số lần (theo `don_vi`) mỗi trang bị bị thiếu, nhiều nhất ở trên."""
    df = pd.DataFrame([{"Trang bị": ten, "Số lần": so} for ten, so in dem.items() if so],
                      columns=["Trang bị", "Số lần"])
    if df.empty:
        return None
    df = df.sort_values("Số lần", ascending=False)
    goc = alt.Chart(df).encode(y=alt.Y("Trang bị:N", sort=df["Trang bị"].tolist(), title=None, axis=AN_TRUC),
                               x=alt.X("Số lần:Q", title=don_vi, axis=alt.Axis(tickMinStep=1, format="d")))
    thanh = goc.mark_bar(cornerRadiusEnd=4, height={"band": 0.55}).encode(
        tooltip=[alt.Tooltip("Trang bị:N"), alt.Tooltip("Số lần:Q", title=don_vi)])
    chu = goc.mark_text(align="left", dx=6, fontSize=12, color="#64748b").encode(text="Số lần:Q")
    return (thanh + chu).properties(height=max(120, 46 * len(df)), padding={"right": 30, "left": 4, "top": 4,
                                                                            "bottom": 4})

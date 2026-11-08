"""AI 輔助之互動式影像處理與人臉分析工具 (Streamlit)
功能:1.影像載入與屬性 2.色彩空間轉換與直方圖 3.自訂卷積核與PSNR 4.人臉偵測
執行:streamlit run app.py
"""
import io
import numpy as np
import cv2
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from PIL.ExifTags import TAGS

st.set_page_config(page_title="影像處理與人臉分析工具", layout="wide")
st.title("🖼️ AI 輔助互動式影像處理與人臉分析工具")


# ---------------------------------------------------------------- 核心演算法
def convolve2d(img, kernel, normalize=True):
    """手動實作 2D 卷積(反射填充,逐通道,向量化)。img: HxW 或 HxWxC uint8"""
    k = np.array(kernel, dtype=np.float32)
    if normalize and abs(k.sum()) > 1e-8:
        k = k / k.sum()                       # 正規化:元素總和為 1,保持亮度
    k = np.flipud(np.fliplr(k))               # 真正的卷積需翻轉核
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    x = img.astype(np.float32)
    if x.ndim == 2:
        x = x[..., None]
    H, W, C = x.shape
    pad = np.pad(x, ((ph, ph), (pw, pw), (0, 0)), mode="reflect")
    out = np.zeros_like(x)
    for i in range(kh):
        for j in range(kw):
            out += k[i, j] * pad[i:i + H, j:j + W, :]
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out[..., 0] if img.ndim == 2 else out


def psnr(a, b):
    """PSNR = 10*log10(MAX^2 / MSE),單位 dB"""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf"), 0.0
    return 10 * np.log10(255.0 ** 2 / mse), mse


def gaussian_kernel(size, sigma=1.0):
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def preset_kernel(name, size):
    if name == "均值模糊 (Box)":
        return np.ones((size, size))
    if name == "高斯模糊 (Gaussian)":
        return gaussian_kernel(size, sigma=size / 3)
    if name == "銳化 (Sharpen)":
        return np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    if name == "Sobel 水平邊緣":
        return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    if name == "Sobel 垂直邊緣":
        return np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    if name == "Laplacian 邊緣":
        return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    if name == "浮雕 (Emboss)":
        return np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    return np.eye(3)


def to_space(rgb, space):
    """回傳 (影像, 通道名稱)。HSV 的 H 範圍為 0~179(OpenCV 慣例)。"""
    if space == "RGB":
        return rgb, ["R", "G", "B"]
    if space == "Grayscale":
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), ["Gray"]
    if space == "HSV":
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), ["H", "S", "V"]
    if space == "YUV":
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV), ["Y", "U", "V"]
    return rgb, ["R", "G", "B"]


def channel_view(img, idx):
    """把單一通道顯示成灰階圖"""
    return img if img.ndim == 2 else img[..., idx]


def plot_hist(img, names, title):
    colors = {"R": "r", "G": "g", "B": "b", "Gray": "k",
              "H": "m", "S": "c", "V": "y", "Y": "k", "U": "b", "Vv": "r"}
    fig, ax = plt.subplots(figsize=(5, 3))
    chans = [img] if img.ndim == 2 else [img[..., i] for i in range(img.shape[2])]
    for ch, n in zip(chans, names):
        h = np.bincount(ch.ravel(), minlength=256)
        ax.plot(h, label=n, color=colors.get(n, None), alpha=0.8)
    ax.set_xlim(0, 255)
    ax.set_xlabel("Pixel value (0-255)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


@st.cache_resource
def load_cascade():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect_faces(rgb, scale, neighbors, min_size):
    """Haar Cascade 人臉偵測;信心度由 levelWeights 經 sigmoid 轉為 0~1"""
    gray = cv2.equalizeHist(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    rects, _, weights = load_cascade().detectMultiScale3(
        gray, scaleFactor=scale, minNeighbors=neighbors,
        minSize=(min_size, min_size), outputRejectLevels=True)
    out = rgb.copy()
    results = []
    for (x, y, w, h), wt in zip(rects, np.ravel(weights)):
        conf = 1 / (1 + np.exp(-float(wt)))
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(out, f"{conf:.2f}", (x, max(y - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        results.append((int(x), int(y), int(w), int(h), conf))
    return out, results


def get_metadata(file_bytes, pil_img, rgb):
    info = {
        "檔案格式": pil_img.format,
        "寬 × 高 (px)": f"{rgb.shape[1]} × {rgb.shape[0]}",
        "色彩模式 (原始)": pil_img.mode,
        "通道數": rgb.shape[2],
        "位元深度": "8 bit / 通道",
        "檔案大小": f"{len(file_bytes) / 1024:.1f} KB",
    }
    try:
        exif = pil_img.getexif()
        for tag_id, val in list(exif.items())[:5]:
            info[f"EXIF:{TAGS.get(tag_id, tag_id)}"] = str(val)[:40]
    except Exception:
        pass
    return info


# ---------------------------------------------------------------- 載入影像
uploaded = st.sidebar.file_uploader("上傳影像 (PNG / JPG / BMP)",
                                    type=["png", "jpg", "jpeg", "bmp"])
if uploaded is None:
    st.info("請從左側上傳一張影像開始。")
    st.stop()

data = uploaded.getvalue()
try:
    pil = Image.open(io.BytesIO(data))
    pil.load()
    rgb = np.array(pil.convert("RGB"))
except Exception as e:
    st.error(f"無法讀取影像:{e}")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(
    ["① 影像屬性", "② 色彩空間與直方圖", "③ 卷積濾波與 PSNR", "④ 人臉偵測"])

# ---------------------------------------------------------------- ①
with tab1:
    c1, c2 = st.columns([2, 1])
    c1.image(rgb, caption="原始影像", use_container_width=True)
    c2.table({k: [v] for k, v in get_metadata(data, pil, rgb).items()}
             if False else
             [{"項目": k, "內容": str(v)} for k, v in get_metadata(data, pil, rgb).items()])

# ---------------------------------------------------------------- ②
with tab2:
    space = st.radio("色彩空間", ["Original", "RGB", "Grayscale", "HSV", "YUV"],
                     horizontal=True)
    if space == "Original":
        st.image(rgb, use_container_width=True)
        img_s, names = rgb, ["R", "G", "B"]
    else:
        img_s, names = to_space(rgb, space)
        cols = st.columns(len(names) + (1 if img_s.ndim == 3 else 0))
        if img_s.ndim == 2:
            cols[0].image(img_s, caption="Grayscale", use_container_width=True)
        else:
            cols[0].image(rgb if space != "RGB" else img_s,
                          caption="原圖", use_container_width=True)
            for i, n in enumerate(names):
                cols[i + 1].image(channel_view(img_s, i), caption=f"{n} 通道",
                                  use_container_width=True)
    st.pyplot(plot_hist(img_s, names, f"Histogram — {space}"))
    if space == "HSV":
        st.caption("註:OpenCV 的 H 值範圍為 0–179。")

# ---------------------------------------------------------------- ③
with tab3:
    left, right = st.columns([1, 2])
    with left:
        mode = st.radio("卷積核來源", ["預設濾波器", "自訂卷積核"])
        normalize = st.checkbox("正規化 (除以核元素總和)", value=True)
        gray_only = st.checkbox("以灰階處理", value=False)
        if mode == "預設濾波器":
            name = st.selectbox("濾波器", ["均值模糊 (Box)", "高斯模糊 (Gaussian)",
                                         "銳化 (Sharpen)", "Sobel 水平邊緣",
                                         "Sobel 垂直邊緣", "Laplacian 邊緣",
                                         "浮雕 (Emboss)"])
            size = st.select_slider("核大小 (僅 Box/Gaussian 可調)", [3, 5, 7, 9], 3)
            kernel = preset_kernel(name, size)
        else:
            txt = st.text_area("輸入卷積核(每列一行,數字以空白或逗號分隔;須為奇數方陣)",
                               "0 -1 0\n-1 5 -1\n0 -1 0", height=130)
            try:
                kernel = np.array([[float(v) for v in line.replace(",", " ").split()]
                                   for line in txt.strip().splitlines()])
                if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1] \
                        or kernel.shape[0] % 2 == 0:
                    raise ValueError("卷積核必須為奇數大小的方陣 (3×3, 5×5, 7×7…)")
            except Exception as e:
                st.error(f"卷積核格式錯誤:{e}")
                st.stop()
        st.write("目前卷積核:")
        st.dataframe(np.round(kernel, 4), hide_index=True)

    src = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if gray_only else rgb
    result = convolve2d(src, kernel, normalize)
    p, mse = psnr(src, result)
    with right:
        a, b = st.columns(2)
        a.image(src, caption="處理前", use_container_width=True)
        b.image(result, caption="處理後", use_container_width=True)
        m1, m2 = st.columns(2)
        m1.metric("PSNR (dB)", "∞ (完全相同)" if np.isinf(p) else f"{p:.2f}")
        m2.metric("MSE", f"{mse:.2f}")
        st.caption("PSNR = 10·log10(255² / MSE);數值越高代表處理後與原圖越相近。")
        st.pyplot(plot_hist(result, ["Gray"] if result.ndim == 2 else ["R", "G", "B"],
                            "處理後直方圖"))

# ---------------------------------------------------------------- ④
with tab4:
    c1, c2, c3 = st.columns(3)
    scale = c1.slider("scaleFactor", 1.05, 1.5, 1.1, 0.05)
    neighbors = c2.slider("minNeighbors", 1, 10, 5)
    min_size = c3.slider("最小人臉尺寸 (px)", 20, 200, 40, 10)
    out, faces = detect_faces(rgb, scale, neighbors, min_size)
    st.image(out, caption=f"偵測到 {len(faces)} 張人臉", use_container_width=True)
    if faces:
        st.table([{"#": i + 1, "x": x, "y": y, "寬": w, "高": h,
                   "信心度": f"{c:.2f}"} for i, (x, y, w, h, c) in enumerate(faces)])
    else:
        st.warning("未偵測到人臉,可嘗試調低 minNeighbors 或最小尺寸。")

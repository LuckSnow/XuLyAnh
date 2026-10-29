import os
import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox
import tkinter as tk
import threading
from single_application import APPLICATION
from photo_fix import run_photo_fix, save_stages
ctk.set_appearance_mode("dark")
BG = "#0f0d18"
SURFACE = "#181522"
SURFACE_2 = "#211d2e"
CANVAS_BG = "#0a0910"
ACCENT = "#ff5fa2"
ACCENT_HOVER = "#ff79b4"
PURPLE = "#9b7bff"
TEXT = "#f7f3fb"
MUTED = "#aaa3b5"
SUCCESS = "#35d0a0"

# ═══════════════════════════════════════════════════════════
#  IMAGE PROCESSING ENGINE  (Tất cả thuật toán từ Demo1-17)
# ═══════════════════════════════════════════════════════════

class IP:
    """Image Processing – các thuật toán xử lý ảnh."""
    @staticmethod
    def to_gray(img):
        if len(img.shape) == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def ensure_bgr(img):
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img
    @staticmethod
    def negative(img, **_):
        """Demo3 – s = 255 – r"""
        return 255 - img

    @staticmethod
    def log_transform(img, c_factor=1.0, **_):
        """Demo4 – s = c·log(1+r)"""
        f = img.astype(np.float32)
        mx = np.max(f)
        if mx == 0:
            return img.copy()
        c = (255.0 / np.log(1 + mx)) * float(c_factor)
        result = c * np.log(1 + f)
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def gamma_correction(img, gamma=1.0, **_):
        """Demo5 – s = r^gamma"""
        f = np.clip(img / 255.0, 0, 1)
        return (np.power(f, float(gamma)) * 255).astype(np.uint8)

    @staticmethod
    def linear_stretch(img, **_):
        """Demo6 – keo gian tuyen tinh"""
        mn, mx = img.min(), img.max()
        if mx == mn:
            return img.copy()
        return ((img.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8)

    @staticmethod
    def brightness_contrast(img, brightness=0, contrast=1.0, **_):
        """s = alpha*r + beta, bão hòa về [0, 255] (không lấy trị tuyệt đối)."""
        result = img.astype(np.float32) * float(contrast) + float(brightness)
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def auto_enhance(img, **_):
        """Tăng tương phản cục bộ trên kênh sáng, giữ màu tự nhiên."""
        return IP.clahe(img, clip=2.0, tile=8)

    @staticmethod
    def rotate(img, angle=0, **_):
        """Xoay quanh tâm và tự mở rộng khung để không cắt mất ảnh."""
        h, w = img.shape[:2]
        center = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        cos_a, sin_a = abs(matrix[0, 0]), abs(matrix[0, 1])
        new_w = max(1, int(h * sin_a + w * cos_a))
        new_h = max(1, int(h * cos_a + w * sin_a))
        matrix[0, 2] += new_w / 2.0 - center[0]
        matrix[1, 2] += new_h / 2.0 - center[1]
        return cv2.warpAffine(img, matrix, (new_w, new_h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(24, 21, 34))

    @staticmethod
    def flip(img, direction=1, **_):
        return cv2.flip(img, int(direction))
    @staticmethod
    def hist_equalize(img, **_):
        """Demo7 – can bang histogram"""
        if len(img.shape) == 3:
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return cv2.equalizeHist(img)

    @staticmethod
    def clahe(img, clip=2.0, tile=8, **_):
        """CLAHE – can bang histogram thich nghi"""
        tile = max(2, int(tile))
        obj = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(tile, tile))
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = obj.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return obj.apply(img)

    # ── Bo loc lam min ────────────────────────────────────────
    @staticmethod
    def mean_filter(img, k=3, **_):
        """Demo8 – loc trung binh"""
        k = max(1, int(k))
        if k % 2 == 0:
            k += 1
        return cv2.blur(img, (k, k))

    @staticmethod
    def gaussian_filter(img, k=3, sigma=1.0, **_):
        """Demo8 – loc Gaussian"""
        k = max(1, int(k))
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(img, (k, k), float(sigma))

    @staticmethod
    def median_filter(img, k=3, **_):
        """Demo8 – loc trung vi"""
        k = max(1, int(k))
        if k % 2 == 0:
            k += 1
        return cv2.medianBlur(img, k)

    @staticmethod
    def sharpen(img, strength=1.0, **_):
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]], np.float32)
        sh = cv2.filter2D(img, -1, kernel)
        s = float(strength)
        return cv2.addWeighted(img, 1 - s, sh, s, 0)

    @staticmethod
    def bilateral(img, d=9, sigma=75, **_):
        return cv2.bilateralFilter(img, int(d), float(sigma), float(sigma))

    # ── Hinh thai hoc ─────────────────────────────────────────
    @staticmethod
    def erosion(img, k=5, **_):
        """Demo9 – phep co"""
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (int(k), int(k)))
        return cv2.erode(img, kern)

    @staticmethod
    def dilation(img, k=5, **_):
        """Demo10 – phep gian"""
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (int(k), int(k)))
        return cv2.dilate(img, kern)

    @staticmethod
    def opening(img, k=5, **_):
        """Demo11 – phep mo (erosion -> dilation)"""
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (int(k), int(k)))
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kern)

    @staticmethod
    def closing(img, k=5, **_):
        """Demo12 – phep dong (dilation -> erosion)"""
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (int(k), int(k)))
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kern)

    # ── Phat hien bien ────────────────────────────────────────
    @staticmethod
    def roberts(img, **_):
        """Demo14 – Roberts edge detection"""
        gray = IP.to_gray(img)
        kx = np.array([[1, 0], [0, -1]], np.float32)
        ky = np.array([[0, 1], [-1, 0]], np.float32)
        Gx = cv2.filter2D(gray, cv2.CV_64F, kx)
        Gy = cv2.filter2D(gray, cv2.CV_64F, ky)
        G = cv2.convertScaleAbs(cv2.magnitude(Gx, Gy))
        return cv2.cvtColor(G, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def prewitt(img, **_):
        """Demo15 – Prewitt edge detection"""
        gray = IP.to_gray(img)
        kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)
        ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], np.float32)
        Gx = cv2.filter2D(gray, cv2.CV_64F, kx)
        Gy = cv2.filter2D(gray, cv2.CV_64F, ky)
        G = cv2.convertScaleAbs(cv2.magnitude(Gx, Gy))
        return cv2.cvtColor(G, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def sobel(img, **_):
        """Demo16 – Sobel edge detection"""
        gray = IP.to_gray(img)
        Gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        Gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        G = cv2.convertScaleAbs(cv2.magnitude(Gx, Gy))
        return cv2.cvtColor(G, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def canny(img, t1=100, t2=200, **_):
        """Demo17 – Canny edge detection (multi-step)"""
        gray = IP.to_gray(img)
        # Theo đúng quy trình trong giáo trình: khử nhiễu Gaussian trước,
        # sau đó gradient, non-maximum suppression và ngưỡng kép trong Canny.
        gray = cv2.GaussianBlur(gray, (5, 5), 1.4)
        low, high = sorted((float(t1), float(t2)))
        edges = cv2.Canny(gray, low, high, L2gradient=True)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # ── Mau sac & Hieu ung ───────────────────────────────────
    @staticmethod
    def grayscale_effect(img, **_):
        return cv2.cvtColor(IP.to_gray(img), cv2.COLOR_GRAY2BGR)

    @staticmethod
    def sepia(img, **_):
        # Công thức sepia chuẩn viết cho RGB; chuyển đổi rõ ràng vì OpenCV là BGR.
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        kernel = np.array([[0.393, 0.769, 0.189],
                           [0.349, 0.686, 0.168],
                           [0.272, 0.534, 0.131]], dtype=np.float32)
        result = cv2.transform(rgb, kernel)
        result = np.clip(result, 0, 255).astype(np.uint8)
        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    @staticmethod
    def hue_saturation(img, hue=0, saturation=1.0, **_):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + float(hue)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * float(saturation), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    @staticmethod
    def temperature(img, value=0, **_):
        result = img.astype(np.float32).copy()
        result[:, :, 0] = np.clip(result[:, :, 0] - float(value), 0, 255)
        result[:, :, 2] = np.clip(result[:, :, 2] + float(value), 0, 255)
        return result.astype(np.uint8)

    @staticmethod
    def vignette(img, strength=0.5, **_):
        h, w = img.shape[:2]
        s = max(0.05, float(strength))
        X = cv2.getGaussianKernel(w, w * (1.0 - s * 0.75))
        Y = cv2.getGaussianKernel(h, h * (1.0 - s * 0.75))
        mask = Y * X.T
        mask = mask / mask.max()
        result = img.copy().astype(np.float64)
        for i in range(3):
            result[:, :, i] *= mask
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def pencil_sketch(img, **_):
        _, sketch = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
        return sketch

    @staticmethod
    def cartoon(img, **_):
        gray = IP.to_gray(img)
        blur = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(blur, 255,
                                      cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(img, 9, 300, 300)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return cv2.bitwise_and(color, edges_bgr)


# ═══════════════════════════════════════════════════════════
#  TOOL REGISTRY
# ═══════════════════════════════════════════════════════════

CATEGORIES = [
    ("sun",  "Co Ban"),
    ("loop", "Bien Doi"),
    ("hist", "Histogram"),
    ("pal",  "Mau Sac"),
    ("wave", "Bo Loc"),
    ("gem",  "Hinh Thai"),
    ("cut",  "Phat Bien"),
    ("star", "Hieu Ung"),
]

CAT_ICONS = ["☀️", "🔄", "📊", "🌈", "🌊", "🔷", "✂️", "✨"]
CAT_NAMES = ["Cơ Bản", "Biến Đổi", "Histogram", "Màu Sắc", "Bộ Lọc", "Hình Thái", "Phát Biên", "Hiệu Ứng"]

CAT_COLORS = [
    "#79c0ff",
    "#d2a8ff",
    "#ffa657",
    "#ff7b72",
    "#56d364",
    "#4dc7fb",
    "#e3b341",
    "#ff79c6",
]

TOOLS = [
    # 0 Co Ban
    {
        "id": "auto_enhance", "cat": 0, "name": "Tự Động Nâng Ảnh",
        "func": IP.auto_enhance, "params": [],
    },
    {
        "id": "brightness_contrast", "cat": 0, "name": "Sáng / Tương Phản",
        "func": IP.brightness_contrast,
        "params": [
            {"key": "brightness", "label": "Độ Sáng",    "min": -100, "max": 100,  "step": 1,    "default": 0,   "is_int": True},
            {"key": "contrast",   "label": "Tương Phản", "min": 0.1,  "max": 3.0,  "step": 0.05, "default": 1.0, "is_int": False},
        ],
    },
    {
        "id": "temperature", "cat": 0, "name": "Nhiệt Độ Màu",
        "func": IP.temperature,
        "params": [
            {"key": "value", "label": "Ấm (+)  /  Lạnh (−)", "min": -80, "max": 80, "step": 1, "default": 0, "is_int": True},
        ],
    },
    {
        "id": "rotate", "cat": 0, "name": "Xoay Ảnh",
        "func": IP.rotate,
        "params": [
            {"key": "angle", "label": "Góc Xoay", "min": -180, "max": 180, "step": 1, "default": 0, "is_int": True},
        ],
    },
    {
        "id": "flip_horizontal", "cat": 0, "name": "Lật Ngang",
        "func": lambda img, **_: IP.flip(img, 1), "params": [],
    },
    {
        "id": "flip_vertical", "cat": 0, "name": "Lật Dọc",
        "func": lambda img, **_: IP.flip(img, 0), "params": [],
    },
    # 1 Bien Doi Diem
    {
        "id": "negative", "cat": 1, "name": "Ảnh Âm Bản",
        "func": IP.negative, "params": [],
    },
    {
        "id": "log_transform", "cat": 1, "name": "Biến Đổi Log",
        "func": IP.log_transform,
        "params": [
            {"key": "c_factor", "label": "Hệ Số C", "min": 0.1, "max": 3.0, "step": 0.1, "default": 1.0, "is_int": False},
        ],
    },
    {
        "id": "gamma_correction", "cat": 1, "name": "Gamma Correction (γ)",
        "func": IP.gamma_correction,
        "params": [
            {"key": "gamma", "label": "Gamma (γ)", "min": 0.1, "max": 5.0, "step": 0.05, "default": 1.0, "is_int": False},
        ],
    },
    {
        "id": "linear_stretch", "cat": 1, "name": "Kéo Giãn Tuyến Tính",
        "func": IP.linear_stretch, "params": [],
    },
    # 2 Histogram
    {
        "id": "hist_equalize", "cat": 2, "name": "Cân Bằng Histogram",
        "func": IP.hist_equalize, "params": [],
    },
    {
        "id": "clahe", "cat": 2, "name": "CLAHE (Nâng Cao)",
        "func": IP.clahe,
        "params": [
            {"key": "clip", "label": "Clip Limit", "min": 1.0, "max": 8.0, "step": 0.5, "default": 2.0, "is_int": False},
            {"key": "tile", "label": "Tile Size",  "min": 2,   "max": 16,  "step": 2,   "default": 8,   "is_int": True},
        ],
    },
    # 3 Mau Sac
    {
        "id": "hue_saturation", "cat": 3, "name": "Màu / Bão Hòa",
        "func": IP.hue_saturation,
        "params": [
            {"key": "hue",        "label": "Màu (Hue)",    "min": -90, "max": 90,  "step": 1,    "default": 0,   "is_int": True},
            {"key": "saturation", "label": "Bão Hòa (Sat)","min": 0.0, "max": 3.0, "step": 0.05, "default": 1.0, "is_int": False},
        ],
    },
    {
        "id": "grayscale_effect", "cat": 3, "name": "Ảnh Xám",
        "func": IP.grayscale_effect, "params": [],
    },
    {
        "id": "sepia", "cat": 3, "name": "Sepia (Hoài Cổ)",
        "func": IP.sepia, "params": [],
    },
    # 4 Bo Loc
    {
        "id": "mean_filter", "cat": 4, "name": "Lọc Trung Bình (Mean)",
        "func": IP.mean_filter,
        "params": [
            {"key": "k", "label": "Kích Thước Kernel", "min": 3, "max": 31, "step": 2, "default": 3, "is_int": True},
        ],
    },
    {
        "id": "gaussian_filter", "cat": 4, "name": "Lọc Gaussian",
        "func": IP.gaussian_filter,
        "params": [
            {"key": "k",     "label": "Kích Thước Kernel", "min": 3,   "max": 31,  "step": 2,   "default": 3,   "is_int": True},
            {"key": "sigma", "label": "Sigma (σ)",          "min": 0.1, "max": 10.0,"step": 0.1, "default": 1.0, "is_int": False},
        ],
    },
    {
        "id": "median_filter", "cat": 4, "name": "Lọc Trung Vị (Median)",
        "func": IP.median_filter,
        "params": [
            {"key": "k", "label": "Kích Thước Kernel", "min": 3, "max": 21, "step": 2, "default": 3, "is_int": True},
        ],
    },
    {
        "id": "sharpen", "cat": 4, "name": "Làm Sắc Nét",
        "func": IP.sharpen,
        "params": [
            {"key": "strength", "label": "Độ Mạnh", "min": 0.0, "max": 2.0, "step": 0.1, "default": 1.0, "is_int": False},
        ],
    },
    {
        "id": "bilateral", "cat": 4, "name": "Bilateral (Làm Mịn Da)",
        "func": IP.bilateral,
        "params": [
            {"key": "d",     "label": "Đường Kính", "min": 3,  "max": 15,  "step": 2, "default": 9,  "is_int": True},
            {"key": "sigma", "label": "Sigma",       "min": 10, "max": 200, "step": 5, "default": 75, "is_int": True},
        ],
    },
    # 5 Hinh Thai Hoc
    {
        "id": "erosion", "cat": 5, "name": "Phép Co (Erosion)",
        "func": IP.erosion,
        "params": [
            {"key": "k", "label": "Kích Thước Kernel", "min": 3, "max": 25, "step": 2, "default": 5, "is_int": True},
        ],
    },
    {
        "id": "dilation", "cat": 5, "name": "Phép Giãn (Dilation)",
        "func": IP.dilation,
        "params": [
            {"key": "k", "label": "Kích Thước Kernel", "min": 3, "max": 25, "step": 2, "default": 5, "is_int": True},
        ],
    },
    {
        "id": "opening", "cat": 5, "name": "Phép Mở (Opening)",
        "func": IP.opening,
        "params": [
            {"key": "k", "label": "Kích Thước Kernel", "min": 3, "max": 25, "step": 2, "default": 5, "is_int": True},
        ],
    },
    {
        "id": "closing", "cat": 5, "name": "Phép Đóng (Closing)",
        "func": IP.closing,
        "params": [
            {"key": "k", "label": "Kích Thước Kernel", "min": 3, "max": 25, "step": 2, "default": 5, "is_int": True},
        ],
    },
    # 6 Phat Bien
    {
        "id": "roberts", "cat": 6, "name": "Roberts Edge",
        "func": IP.roberts, "params": [],
    },
    {
        "id": "prewitt", "cat": 6, "name": "Prewitt Edge",
        "func": IP.prewitt, "params": [],
    },
    {
        "id": "sobel", "cat": 6, "name": "Sobel Edge",
        "func": IP.sobel, "params": [],
    },
    {
        "id": "canny", "cat": 6, "name": "Canny Edge",
        "func": IP.canny,
        "params": [
            {"key": "t1", "label": "Ngưỡng Thấp", "min": 0, "max": 255, "step": 5, "default": 100, "is_int": True},
            {"key": "t2", "label": "Ngưỡng Cao",  "min": 0, "max": 255, "step": 5, "default": 200, "is_int": True},
        ],
    },
    # 7 Hieu Ung
    {
        "id": "vignette", "cat": 7, "name": "Vignette (Viền Mờ)",
        "func": IP.vignette,
        "params": [
            {"key": "strength", "label": "Độ Mạnh", "min": 0.05, "max": 0.95, "step": 0.05, "default": 0.5, "is_int": False},
        ],
    },
    {
        "id": "pencil_sketch", "cat": 7, "name": "Phác Thảo Bút Chì",
        "func": IP.pencil_sketch, "params": [],
    },
    {
        "id": "cartoon", "cat": 7, "name": "Hoạt Hình (Cartoon)",
        "func": IP.cartoon, "params": [],
    },
]

VISIBLE_CATEGORIES = (1, 2, 4, 5, 6, 8)
CAT_ICONS.append("✨")
CAT_NAMES.append("Ứng Dụng")
CAT_COLORS.append("#ff79c6")
TOOLS = [tool for tool in TOOLS if tool["cat"] in VISIBLE_CATEGORIES]
TOOLS.append({"id": "single_application", "cat": 8,
              "name": APPLICATION.name, "func": APPLICATION.process,
              "params": list(APPLICATION.params)})

TOOL_THEORY = {
    "single_application": APPLICATION.description,
    "auto_enhance": "CLAHE cân bằng histogram cục bộ có giới hạn tương phản.",
    "brightness_contrast": "Biến đổi tuyến tính: s = αr + β.",
    "temperature": "Điều chỉnh cân bằng kênh đỏ và xanh dương.",
    "rotate": "Biến đổi hình học affine quanh tâm ảnh.",
    "flip_horizontal": "Phép đối xứng ảnh qua trục dọc.",
    "flip_vertical": "Phép đối xứng ảnh qua trục ngang.",
    "negative": "Biến đổi âm bản 8-bit: s = 255 − r.",
    "log_transform": "Biến đổi log: s = c·log(1+r), mở rộng vùng tối.",
    "gamma_correction": "Biến đổi lũy thừa: s = 255·(r/255)^γ.",
    "linear_stretch": "Ánh xạ [rmin, rmax] về toàn dải [0, 255].",
    "hist_equalize": "Cân bằng histogram bằng hàm phân phối tích lũy CDF.",
    "clahe": "Cân bằng histogram thích nghi theo từng ô ảnh.",
    "mean_filter": "Lọc thông thấp bằng trung bình lân cận.",
    "gaussian_filter": "Tích chập với kernel Gaussian để giảm nhiễu.",
    "median_filter": "Thay pixel bằng trung vị; hiệu quả với nhiễu muối tiêu.",
    "sharpen": "Lọc thông cao làm nổi chi tiết và đường biên.",
    "bilateral": "Làm mịn theo không gian và cường độ, giúp giữ biên.",
    "erosion": "Phép co hình thái với phần tử cấu trúc hình chữ nhật.",
    "dilation": "Phép giãn hình thái với phần tử cấu trúc hình chữ nhật.",
    "opening": "Phép mở = co rồi giãn; loại chi tiết sáng nhỏ.",
    "closing": "Phép đóng = giãn rồi co; lấp khe/lỗ tối nhỏ.",
    "roberts": "Gradient bậc nhất theo hai hướng chéo với mặt nạ 2×2.",
    "prewitt": "Gradient bậc nhất X/Y với hai mặt nạ 3×3.",
    "sobel": "Gradient có trọng số trung tâm, giảm nhiễu nhẹ.",
    "canny": "Gaussian → gradient → cực đại cục bộ → ngưỡng kép → nối biên.",
    "hue_saturation": "Biến đổi trong không gian màu HSV.",
    "grayscale_effect": "Chiếu RGB/BGR về một kênh độ chói.",
    "sepia": "Biến đổi tuyến tính ba kênh tạo tông nâu cổ điển.",
    "vignette": "Nhân ảnh với mặt nạ Gaussian hai chiều.",
    "pencil_sketch": "Làm mịn giữ biên và ánh xạ sắc độ thành nét chì.",
    "cartoon": "Bilateral giữ màu kết hợp ngưỡng thích nghi tạo đường nét.",
}


# ═══════════════════════════════════════════════════════════
#  MAIN APPLICATION WINDOW
# ═══════════════════════════════════════════════════════════

class PhotoEditorApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("Lumière Studio  •  Trình chỉnh sửa ảnh")
        self.geometry("1560x950")
        self.minsize(1200, 720)
        self.configure(fg_color=BG)

        # State
        self.cv_image       = None
        self.original_image = None
        self.history        = []
        self.redo_stack     = []
        self.selected_tool  = None
        self.current_cat    = 0
        self.param_vars     = {}
        self.tk_image       = None
        self.zoom           = 1.0
        self._preview_job   = None
        self._tool_btns     = {}
        self._preview_image = None

        self._build_layout()
        self._build_topbar()
        self._build_left()
        self._build_canvas()
        self._build_right()

        self.bind("<Control-z>", lambda e: self.undo())
        self.bind("<Control-y>", lambda e: self.redo())
        self.bind("<Control-o>", lambda e: self.open_image())
        self.bind("<Control-s>", lambda e: self.save_image())
        self.bind("<Control-r>", lambda e: self.reset_image())

    # ────────────────────────────────────────────────────────
    # LAYOUT
    # ────────────────────────────────────────────────────────

    def _build_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

    # ────────────────────────────────────────────────────────
    # TOP BAR
    # ────────────────────────────────────────────────────────

    def _build_topbar(self):
        bar = ctk.CTkFrame(self, fg_color=SURFACE, height=76, corner_radius=0)
        bar.grid(row=0, column=0, columnspan=3, sticky="ew")
        bar.grid_propagate(False)
        bar.grid_columnconfigure(2, weight=1)

        ctk.CTkLabel(
            bar,
            text="✦  LUMIÈRE",
            font=ctk.CTkFont("Arial", 22, "bold"),
            text_color=ACCENT,
        ).grid(row=0, column=0, padx=22, pady=12, sticky="w")

        ctk.CTkLabel(bar, text="STUDIO", font=ctk.CTkFont("Arial", 10, "bold"),
                     text_color=MUTED).grid(row=0, column=1, padx=2)

        btn_frame = ctk.CTkFrame(bar, fg_color="transparent")
        btn_frame.grid(row=0, column=2, pady=12, padx=12, sticky="w")

        for txt, cmd, fg, hv in [
            ("＋  Mở ảnh",    self.open_image,  ACCENT, ACCENT_HOVER),
            ("⇩  Xuất ảnh",   self.save_image,  PURPLE, "#ad94ff"),
            ("↶  Hoàn tác",   self.undo,        SURFACE_2, "#302a40"),
            ("↷  Làm lại",    self.redo,        SURFACE_2, "#302a40"),
            ("⟳  Ảnh gốc",    self.reset_image, SURFACE_2, "#302a40"),
        ]:
            ctk.CTkButton(
                btn_frame, text=txt, command=cmd,
                fg_color=fg, hover_color=hv,
                width=112, height=40, corner_radius=12,
                font=ctk.CTkFont("Arial", 12, "bold"),
            ).pack(side="left", padx=4)

        self.info_label = ctk.CTkLabel(
            bar,
            text="Chưa có ảnh  •  Ctrl+O để mở",
            font=ctk.CTkFont("Arial", 11),
            text_color=MUTED,
        )
        self.info_label.grid(row=0, column=3, padx=20, sticky="e")

    # ────────────────────────────────────────────────────────
    # LEFT PANEL
    # ────────────────────────────────────────────────────────

    def _build_left(self):
        left = ctk.CTkFrame(self, fg_color=SURFACE, width=244, corner_radius=0)
        left.grid(row=1, column=0, sticky="nsew")
        left.grid_propagate(False)

        ctk.CTkLabel(
            left, text="KHÔNG GIAN SÁNG TẠO",
            font=ctk.CTkFont("Arial", 10, "bold"),
            text_color=MUTED,
        ).pack(anchor="w", padx=16, pady=(18, 6))

        self._cat_btns = {}
        for i in VISIBLE_CATEGORIES:
            name = CAT_NAMES[i]
            icon = CAT_ICONS[i]
            btn = ctk.CTkButton(
                left,
                text=f"{icon}  {name}",
                command=lambda idx=i: self._select_cat(idx),
                fg_color="transparent",
                hover_color=SURFACE_2,
                text_color=CAT_COLORS[i],
                anchor="w",
                width=224,
                height=42,
                corner_radius=12,
                font=ctk.CTkFont("Arial", 13),
            )
            btn.pack(padx=8, pady=1)
            self._cat_btns[i] = btn

        ctk.CTkFrame(left, fg_color=SURFACE_2, height=1).pack(fill="x", padx=12, pady=10)

        ctk.CTkLabel(
            left, text="CÔNG CỤ",
            font=ctk.CTkFont("Arial", 10, "bold"),
            text_color=MUTED,
        ).pack(anchor="w", padx=16, pady=(0, 6))

        self._tool_list = ctk.CTkScrollableFrame(left, fg_color="transparent")
        self._tool_list.pack(fill="both", expand=True, padx=4, pady=(0, 8))

        self._select_cat(VISIBLE_CATEGORIES[0])

    def _select_cat(self, idx):
        self.current_cat = idx
        for i, btn in self._cat_btns.items():
            btn.configure(fg_color=SURFACE_2 if i == idx else "transparent")

        for w in self._tool_list.winfo_children():
            w.destroy()
        self._tool_btns = {}

        for tool in [t for t in TOOLS if t["cat"] == idx]:
            btn = ctk.CTkButton(
                self._tool_list,
                text=f"   {tool['name']}",
                command=lambda tid=tool["id"]: self._select_tool(tid),
                fg_color="transparent",
                hover_color=SURFACE_2,
                text_color=TEXT,
                anchor="w",
                width=208,
                height=34,
                corner_radius=10,
                font=ctk.CTkFont("Arial", 12),
            )
            btn.pack(padx=4, pady=1)
            self._tool_btns[tool["id"]] = btn

    # ────────────────────────────────────────────────────────
    # CANVAS
    # ────────────────────────────────────────────────────────

    def _build_canvas(self):
        wrap = ctk.CTkFrame(self, fg_color=CANVAS_BG, corner_radius=0)
        wrap.grid(row=1, column=1, sticky="nsew")
        wrap.grid_rowconfigure(0, weight=1)
        wrap.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(wrap, bg=CANVAS_BG, highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.canvas.bind("<Configure>",  lambda e: self._refresh_canvas())
        self.canvas.bind("<MouseWheel>", self._on_scroll)
        self.canvas.bind("<Button-4>",   self._on_scroll)
        self.canvas.bind("<Button-5>",   self._on_scroll)
        # Giữ phím Space hoặc nút "Ảnh gốc" để so sánh trước/sau.
        self.bind("<KeyPress-space>", self._show_original)
        self.bind("<KeyRelease-space>", self._show_edited)

        self.zoom_label = ctk.CTkLabel(
            wrap, text="100%",
            fg_color=SURFACE_2, corner_radius=8,
            font=ctk.CTkFont("Arial", 11),
            text_color=MUTED, padx=10, pady=4,
        )
        self.zoom_label.place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-10)

        zoom_row = ctk.CTkFrame(wrap, fg_color=SURFACE_2, corner_radius=10)
        zoom_row.place(relx=0.0, rely=1.0, anchor="sw", x=10, y=-10)
        for txt, d in [("−", -0.25), ("+", 0.25)]:
            ctk.CTkButton(
                zoom_row, text=txt, width=34, height=30,
                corner_radius=6,
                fg_color="transparent", hover_color="#30363d",
                font=ctk.CTkFont("Arial", 16, "bold"),
                command=lambda dz=d: self._zoom_step(dz),
            ).pack(side="left", padx=3, pady=4)
        ctk.CTkButton(
            zoom_row, text="R", width=34, height=30,
            corner_radius=6,
            fg_color="transparent", hover_color="#30363d",
            font=ctk.CTkFont("Arial", 13, "bold"),
            command=self._zoom_reset,
        ).pack(side="left", padx=3, pady=4)

        self.compare_btn = ctk.CTkButton(
            wrap, text="◐  Giữ để xem ảnh gốc", width=170, height=36,
            corner_radius=12, fg_color=SURFACE_2, hover_color="#302a40",
            text_color=TEXT, font=ctk.CTkFont("Arial", 11, "bold"),
        )
        self.compare_btn.place(relx=0.5, rely=1.0, anchor="s", y=-10)
        self.compare_btn.bind("<ButtonPress-1>", self._show_original)
        self.compare_btn.bind("<ButtonRelease-1>", self._show_edited)

        self._show_placeholder()

    # ────────────────────────────────────────────────────────
    # RIGHT PANEL
    # ────────────────────────────────────────────────────────

    def _build_right(self):
        right = ctk.CTkFrame(self, fg_color=SURFACE, width=318, corner_radius=0)
        right.grid(row=1, column=2, sticky="nsew")
        right.grid_propagate(False)

        ctk.CTkLabel(
            right, text="TINH CHỈNH",
            font=ctk.CTkFont("Arial", 10, "bold"),
            text_color=MUTED,
        ).pack(anchor="w", padx=16, pady=(18, 6))

        self._ctrl_frame = ctk.CTkScrollableFrame(right, fg_color="transparent", height=340)
        self._ctrl_frame.pack(fill="x", padx=8, pady=4)
        self._show_no_tool_hint()

        self._preview_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            right,
            text="Xem trước theo thời gian thực",
            variable=self._preview_var,
            font=ctk.CTkFont("Arial", 12),
            text_color=MUTED,
            hover_color=ACCENT_HOVER,
            checkmark_color=ACCENT,
            command=self._toggle_preview,
        ).pack(anchor="w", padx=14, pady=(6, 0))

        self._apply_btn = ctk.CTkButton(
            right,
            text="✓  Áp dụng chỉnh sửa",
            command=self.apply_tool,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            height=48, corner_radius=14,
            font=ctk.CTkFont("Arial", 14, "bold"),
            state="disabled",
        )
        self._apply_btn.pack(fill="x", padx=14, pady=10)

        ctk.CTkFrame(right, fg_color=SURFACE_2, height=1).pack(fill="x", padx=12, pady=4)

        ctk.CTkLabel(
            right, text="HISTOGRAM",
            font=ctk.CTkFont("Arial", 10, "bold"),
            text_color=MUTED,
        ).pack(anchor="w", padx=16, pady=(6, 4))

        hist_wrap = ctk.CTkFrame(right, fg_color=BG, corner_radius=12, height=135)
        hist_wrap.pack(fill="x", padx=12, pady=4)
        hist_wrap.pack_propagate(False)

        self._hist_cv = tk.Canvas(hist_wrap, bg=BG, highlightthickness=0)
        self._hist_cv.pack(fill="both", expand=True, padx=4, pady=4)

        ctk.CTkFrame(right, fg_color=SURFACE_2, height=1).pack(fill="x", padx=12, pady=6)

        self._stat_label = ctk.CTkLabel(
            right, text="",
            font=ctk.CTkFont("Arial", 10),
            text_color="#6e7681",
            justify="left",
        )
        self._stat_label.pack(anchor="w", padx=16, pady=4)

    # ────────────────────────────────────────────────────────
    # TOOL SELECTION
    # ────────────────────────────────────────────────────────

    def _select_tool(self, tool_id):
        for btn in self._tool_btns.values():
            btn.configure(fg_color="transparent")

        self.selected_tool = tool_id

        if tool_id in self._tool_btns:
            self._tool_btns[tool_id].configure(fg_color="#1f3b6b", text_color=CAT_COLORS[self.current_cat])

        self._build_controls(tool_id)

        if self.cv_image is not None:
            self._apply_btn.configure(state="normal")
            if self._preview_var.get():
                self._trigger_preview()

    def _build_controls(self, tool_id):
        for w in self._ctrl_frame.winfo_children():
            w.destroy()
        self.param_vars = {}

        tool = next((t for t in TOOLS if t["id"] == tool_id), None)
        if not tool:
            return

        cat_color = CAT_COLORS[tool["cat"]]
        cat_name  = CAT_NAMES[tool["cat"]]
        cat_icon  = CAT_ICONS[tool["cat"]]

        # Header card
        card = ctk.CTkFrame(self._ctrl_frame, fg_color=SURFACE_2, corner_radius=12)
        card.pack(fill="x", padx=4, pady=(4, 10))

        ctk.CTkLabel(card, text=cat_icon, font=ctk.CTkFont("Arial", 20),
                     text_color=cat_color).pack(side="left", padx=12, pady=10)
        info = ctk.CTkFrame(card, fg_color="transparent")
        info.pack(side="left", fill="both", expand=True, pady=10)
        ctk.CTkLabel(info, text=tool["name"],
                     font=ctk.CTkFont("Arial", 13, "bold"),
                     text_color=cat_color, anchor="w").pack(anchor="w")
        ctk.CTkLabel(info, text=cat_name,
                     font=ctk.CTkFont("Arial", 11),
                     text_color="#6e7681", anchor="w").pack(anchor="w")

        theory = TOOL_THEORY.get(tool_id, "Hiệu ứng xử lý ảnh số.")
        ctk.CTkLabel(
            self._ctrl_frame, text=theory, wraplength=260, justify="left",
            font=ctk.CTkFont("Arial", 11), text_color=MUTED,
        ).pack(fill="x", padx=8, pady=(0, 8))

        if tool["params"]:
            for p in tool["params"]:
                self._make_slider(p)
        else:
            ctk.CTkLabel(
                self._ctrl_frame,
                text="Cong cu nay khong co tham so\nNhan \"Ap Dung\" de xu ly anh",
                font=ctk.CTkFont("Arial", 12),
                text_color="#6e7681",
                justify="center",
            ).pack(pady=20)

    def _make_slider(self, p):
        box = ctk.CTkFrame(self._ctrl_frame, fg_color=BG, corner_radius=10)
        box.pack(fill="x", padx=4, pady=5)

        top_row = ctk.CTkFrame(box, fg_color="transparent")
        top_row.pack(fill="x", padx=10, pady=(8, 2))

        ctk.CTkLabel(top_row, text=p["label"],
                     font=ctk.CTkFont("Arial", 12),
                     text_color="#c9d1d9", anchor="w").pack(side="left")

        var = ctk.DoubleVar(value=p["default"])
        self.param_vars[p["key"]] = var

        val_lbl = ctk.CTkLabel(top_row,
                               text=self._fmt(p, p["default"]),
                               font=ctk.CTkFont("Arial", 12, "bold"),
                               text_color="#58a6ff", anchor="e", width=55)
        val_lbl.pack(side="right")

        steps = max(1, int(round((p["max"] - p["min"]) / p["step"])))
        ctk.CTkSlider(
            box,
            from_=p["min"], to=p["max"],
            variable=var,
            number_of_steps=steps,
            progress_color=ACCENT,
            button_color="#ff9ac5",
            button_hover_color=ACCENT_HOVER,
        ).pack(fill="x", padx=10, pady=(0, 4))

        def _on_change(lbl=val_lbl, param=p):
            lbl.configure(text=self._fmt(param, float(var.get())))
            if self._preview_var.get():
                self._trigger_preview()

        var.trace_add("write", lambda *_: _on_change())

        def _reset(pv=p["default"], v=var):
            v.set(pv)

        ctk.CTkButton(box, text="Reset", command=_reset,
                      fg_color="transparent", hover_color="#21262d",
                      text_color="#6e7681", width=48, height=20,
                      font=ctk.CTkFont("Arial", 10),
                      corner_radius=4).pack(anchor="e", padx=8, pady=(0, 6))

    @staticmethod
    def _fmt(param, val):
        return str(int(round(val))) if param["is_int"] else f"{val:.2f}"

    def _show_no_tool_hint(self):
        for w in self._ctrl_frame.winfo_children():
            w.destroy()
        ctk.CTkLabel(
            self._ctrl_frame,
            text="<- Chon nhom cong cu\n    roi chon cong cu cu the",
            font=ctk.CTkFont("Arial", 12),
            text_color="#6e7681",
            justify="left",
        ).pack(pady=28)

    # ────────────────────────────────────────────────────────
    # IMAGE OPERATIONS
    # ────────────────────────────────────────────────────────

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Mo anh",
            filetypes=[
                ("Anh", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp"),
                ("Tat ca", "*.*"),
            ],
        )
        if not path:
            return
        # imdecode/fromfile hỗ trợ tốt đường dẫn Unicode trên Windows.
        try:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except (OSError, ValueError, cv2.error):
            img = None
        if img is None:
            messagebox.showerror("Loi", "Khong the doc file anh!\nThu file khac.")
            return

        self.cv_image       = img
        self.original_image = img.copy()
        self.history        = [img.copy()]
        self.redo_stack     = []
        self.zoom           = 1.0

        h, w = img.shape[:2]
        name = os.path.basename(path)
        self.info_label.configure(
            text=f"{name}   |   {w} x {h} px   |   Ctrl+S luu"
        )
        self._update_stats()

        if self.selected_tool:
            self._apply_btn.configure(state="normal")

        self._refresh_canvas()
        self._update_histogram()

    def save_image(self):
        if self.cv_image is None:
            messagebox.showwarning("Canh bao", "Chua co anh de luu!")
            return
        path = filedialog.asksaveasfilename(
            title="Luu anh",
            defaultextension=".jpg",
            filetypes=[
                ("JPEG", "*.jpg"), ("PNG", "*.png"),
                ("BMP", "*.bmp"), ("TIFF", "*.tiff"),
            ],
        )
        if path:
            ext = os.path.splitext(path)[1].lower() or ".jpg"
            params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in (".jpg", ".jpeg") else []
            ok, encoded = cv2.imencode(ext, self.cv_image, params)
            if not ok:
                messagebox.showerror("Lỗi", "Không thể mã hóa định dạng ảnh này.")
                return
            encoded.tofile(path)
            messagebox.showinfo("Thành công", f"Đã xuất ảnh chất lượng cao:\n{os.path.basename(path)}")

    def undo(self):
        if len(self.history) <= 1:
            return
        self.redo_stack.append(self.history.pop())
        self.cv_image = self.history[-1].copy()
        self._preview_image = None
        self._refresh_canvas()
        self._update_histogram()
        self._update_stats()

    def redo(self):
        if not self.redo_stack:
            return
        img = self.redo_stack.pop()
        self.history.append(img)
        self.cv_image = img.copy()
        self._preview_image = None
        self._refresh_canvas()
        self._update_histogram()
        self._update_stats()

    def reset_image(self):
        if self.original_image is None:
            return
        self.cv_image   = self.original_image.copy()
        self.history    = [self.cv_image.copy()]
        self.redo_stack = []
        self._preview_image = None
        self._refresh_canvas()
        self._update_histogram()
        self._update_stats()

    def apply_tool(self):
        if self.selected_tool is None or self.cv_image is None:
            return
        tool = next((t for t in TOOLS if t["id"] == self.selected_tool), None)
        if not tool:
            return
        params = self._get_params(tool)
        if self.selected_tool == "single_application":
            source = self.cv_image.copy()
            self._apply_btn.configure(state="disabled", text="Đang xử lý PhotoFix...")

            def work():
                try:
                    result = run_photo_fix(source, **params)
                    paths = save_stages(result)
                    self.after(0, lambda: finish(result.final, paths, None))
                except Exception as exc:
                    self.after(0, lambda error=str(exc): finish(None, None, error))

            def finish(image, paths, error):
                self._apply_btn.configure(state="normal", text="✓  Áp dụng chỉnh sửa")
                if error:
                    messagebox.showerror("Lỗi PhotoFix", error)
                    return
                self.cv_image = image
                self.history.append(image.copy())
                self.redo_stack = []
                self._preview_image = None
                self._refresh_canvas()
                self._update_histogram()
                self._update_stats()
                self.info_label.configure(text=f"PhotoFix hoàn tất • Đã lưu: {paths['final'].name}")

            threading.Thread(target=work, daemon=True).start()
            return
        try:
            result = tool["func"](self.cv_image, **params)
            result = IP.ensure_bgr(result)
            self.cv_image = result
            self.history.append(result.copy())
            self.redo_stack = []
            self._preview_image = None
            self._refresh_canvas()
            self._update_histogram()
            self._update_stats()
        except Exception as e:
            messagebox.showerror("Loi xu ly", str(e))

    def _get_params(self, tool):
        params = {}
        for p in tool["params"]:
            raw = self.param_vars[p["key"]].get()
            params[p["key"]] = int(round(raw)) if p["is_int"] else float(raw)
        return params

    # ────────────────────────────────────────────────────────
    # PREVIEW (debounced)
    # ────────────────────────────────────────────────────────

    def _trigger_preview(self):
        if self.selected_tool == "single_application":
            return
        if self._preview_job:
            self.after_cancel(self._preview_job)
        self._preview_job = self.after(160, self._do_preview)

    def _do_preview(self):
        if self.selected_tool is None or self.cv_image is None:
            return
        tool = next((t for t in TOOLS if t["id"] == self.selected_tool), None)
        if not tool:
            return
        params = self._get_params(tool)
        try:
            result = tool["func"](self.cv_image, **params)
            result = IP.ensure_bgr(result)
            self._preview_image = result
            self._refresh_canvas(result)
        except Exception:
            self._refresh_canvas()

    def _toggle_preview(self):
        if self._preview_var.get():
            self._trigger_preview()
        else:
            self._preview_image = None
            self._refresh_canvas()

    def _show_original(self, _event=None):
        if self.original_image is not None:
            self._refresh_canvas(self.original_image)

    def _show_edited(self, _event=None):
        display = self._preview_image if self._preview_var.get() else None
        self._refresh_canvas(display)

    def _update_stats(self):
        if self.cv_image is None:
            return
        h, w = self.cv_image.shape[:2]
        gray = IP.to_gray(self.cv_image)
        self._stat_label.configure(
            text=(f"Kích thước  {w} × {h} px\n"
                  f"Độ sáng TB  {gray.mean():.1f}   •   Độ lệch  {gray.std():.1f}\n"
                  f"Miền mức xám  {int(gray.min())} — {int(gray.max())}")
        )

    # ────────────────────────────────────────────────────────
    # DISPLAY
    # ────────────────────────────────────────────────────────

    def _show_placeholder(self):
        cw = self.canvas.winfo_width()  or 900
        ch = self.canvas.winfo_height() or 640
        self.canvas.delete("all")
        for x in range(0, cw, 80):
            self.canvas.create_line(x, 0, x, ch, fill="#0a0f16", width=1)
        for y in range(0, ch, 80):
            self.canvas.create_line(0, y, cw, y, fill="#0a0f16", width=1)
        cx, cy = cw // 2, ch // 2
        self.canvas.create_text(cx, cy - 28, text="✦",
                                font=("Arial", 36, "bold"), fill=ACCENT)
        self.canvas.create_text(cx, cy + 18, text="Mở một bức ảnh để bắt đầu sáng tạo",
                                font=("Arial", 17, "bold"), fill="#6e6578")
        self.canvas.create_text(cx, cy + 24,
                                text="",
                                font=("Arial", 12), fill="#493f55")
        self.canvas.create_text(cx, cy + 54,
                                text="JPG · PNG · BMP · TIFF · WEBP   •   Ctrl+O để mở ảnh",
                                font=("Arial", 11), fill="#493f55")

    def _refresh_canvas(self, img=None):
        display = img if img is not None else self.cv_image
        if display is None:
            self._show_placeholder()
            return

        cw = self.canvas.winfo_width()  or 900
        ch = self.canvas.winfo_height() or 640
        h, w = display.shape[:2]

        scale = min(cw / w, ch / h) * self.zoom
        nw    = max(1, int(w * scale))
        nh    = max(1, int(h * scale))

        rgb     = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).resize((nw, nh), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")

        # Subtle checkerboard
        sq = 18
        for row in range(0, ch, sq):
            for col in range(0, cw, sq):
                c = "#0d0b13" if (row // sq + col // sq) % 2 == 0 else "#110e19"
                self.canvas.create_rectangle(col, row, col + sq, row + sq, fill=c, outline="")

        self.canvas.create_image(cw // 2, ch // 2, anchor="center", image=self.tk_image)
        self.zoom_label.configure(text=f"{int(self.zoom * 100)}%")

    def _update_histogram(self):
        if self.cv_image is None:
            return
        try:
            cv = self._hist_cv
            hw = cv.winfo_width()  or 268
            hh = cv.winfo_height() or 120
            cv.delete("all")

            if len(self.cv_image.shape) == 3:
                colors = ["#4dc7fb", "#56d364", "#ff7b72"]
                for ch_idx, col in enumerate(colors):
                    hist = cv2.calcHist([self.cv_image], [ch_idx], None, [256], [0, 256]).flatten()
                    mx   = hist.max()
                    if mx == 0:
                        continue
                    pts = []
                    for i in range(256):
                        pts += [int(i * hw / 256), hh - int(hist[i] / mx * (hh - 6)) - 2]
                    if len(pts) >= 4:
                        cv.create_line(pts, fill=col, width=1, smooth=True)
            else:
                hist = cv2.calcHist([self.cv_image], [0], None, [256], [0, 256]).flatten()
                mx   = hist.max()
                if mx > 0:
                    pts = []
                    for i in range(256):
                        pts += [int(i * hw / 256), hh - int(hist[i] / mx * (hh - 6)) - 2]
                    if len(pts) >= 4:
                        cv.create_line(pts, fill="#58a6ff", width=1, smooth=True)
        except Exception:
            pass

    # ────────────────────────────────────────────────────────
    # ZOOM
    # ────────────────────────────────────────────────────────

    def _on_scroll(self, event):
        if hasattr(event, "delta"):
            factor = 1.1 if event.delta > 0 else 0.9
        else:
            factor = 1.1 if event.num == 4 else 0.9
        self.zoom = max(0.05, min(16.0, self.zoom * factor))
        self._refresh_canvas()

    def _zoom_step(self, d):
        self.zoom = max(0.05, min(16.0, self.zoom + d))
        self._refresh_canvas()

    def _zoom_reset(self):
        self.zoom = 1.0
        self._refresh_canvas()


# ═══════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = PhotoEditorApp()
    app.mainloop()

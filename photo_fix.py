"""PhotoFix: conservative old-photo repair and local DDColor colorization.

Input and output are uint8 OpenCV BGR images. AI colors are plausible guesses,
not historical evidence. Missing image content needs a mask or specialist model.
"""

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time
from urllib.request import urlopen

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "ddcolor_paper_tiny_w16.onnx"
MODEL_URL = ("https://huggingface.co/openpixels-app/browser-models/resolve/main/"
             "ddcolor_paper_tiny_w16.onnx")
MODEL_BYTES = 115_401_645
MODEL_SHA256 = "87854113d29989a6fc3c680b92b21f2cd5377a54b5babd0c5d2a53469f59244a"
MAX_PIXELS = 24_000_000


@dataclass
class PhotoFixResult:
    final: np.ndarray
    stages: dict[str, np.ndarray]
    assessment: dict[str, float | bool]
    elapsed_seconds: float
    colorized: bool


def validate_image(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Ảnh phải là mảng NumPy uint8.")
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        bgr = image.copy()
    elif image.ndim == 3 and image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        raise ValueError("Chỉ hỗ trợ ảnh xám, BGR hoặc BGRA 8-bit.")
    height, width = bgr.shape[:2]
    if height < 16 or width < 16:
        raise ValueError("Ảnh phải rộng và cao tối thiểu 16 pixel.")
    if height * width > MAX_PIXELS:
        raise ValueError("Ảnh vượt 24 triệu pixel; hãy giảm kích thước trước khi xử lý.")
    return bgr


def assess_image(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    light = lab[:, :, 0]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, 3)
    # A small inter-channel difference means a genuinely neutral grayscale scan.
    channel_difference = np.max(bgr, axis=2).astype(np.int16) - np.min(bgr, axis=2)
    grayscale = float(np.percentile(channel_difference, 90)) <= 5.0
    p5, p95 = np.percentile(light, [5, 95])
    chroma = lab[:, :, 1:].reshape(-1, 2)
    median_a, median_b = np.median(chroma, axis=0)
    spread_a, spread_b = np.std(chroma, axis=0)
    sepia = bool(not grayscale and p95 - p5 > 65 and
                 132 <= median_a <= 155 and 138 <= median_b <= 170 and
                 spread_a < 7 and spread_b < 9)
    noise = float(np.median(np.abs(gray.astype(np.int16) - median.astype(np.int16))))
    local_delta = np.abs(gray.astype(np.int16) - median.astype(np.int16))
    impulse_fraction = float(np.mean(((gray < 8) | (gray > 247)) & (local_delta > 40)))
    return {
        "grayscale": grayscale,
        "sepia": sepia,
        "median_light": float(np.median(light)),
        "light_range": float(p95 - p5),
        "noise": noise,
        "impulse_fraction": impulse_fraction,
    }


def find_bright_damage(bgr, strength=1.0, sepia=None):
    """Find only compact bright spots; broad missing areas need a supplied mask."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    residue = cv2.subtract(gray, opened)
    level = int(np.clip(39 - 9 * float(strength), 19, 42))
    candidates = np.uint8((residue > level) & (gray > 220)) * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, 8)
    mask = np.zeros_like(gray)
    max_area = max(12, int(gray.size * 0.0002))
    for index in range(1, count):
        _, _, width, height, area = stats[index]
        small_spot = area <= max_area and width <= 14 and height <= 14
        thin_scratch = (min(width, height) <= 4 and
                        max(width, height) <= 0.6 * max(gray.shape) and
                        area <= max(500, int(gray.size * 0.005)))
        if small_spot or thin_scratch:
            mask[labels == index] = 255
    if sepia is None:
        sepia = assess_image(bgr)["sepia"]
    if sepia:
        # White paper loss loses the uniform brown chroma of a sepia print.
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lum, a, b = cv2.split(lab)
        cast_a, cast_b = float(np.median(a)), float(np.median(b))
        neutrality = (np.abs(a.astype(np.int16) - 128) +
                      np.abs(b.astype(np.int16) - 128)) < 20
        cast_distance = (np.abs(a.astype(np.float32) - cast_a) +
                         np.abs(b.astype(np.float32) - cast_b)) > 14
        paper_loss = np.uint8((lum > 165) & neutrality & cast_distance) * 255
        mask = cv2.bitwise_or(mask, paper_loss)
    return cv2.dilate(mask, np.ones((3, 3), np.uint8))


def repair_damage(bgr, strength=1.0, damage_mask=None, sepia=None):
    mask = find_bright_damage(bgr, strength, sepia=sepia)
    if damage_mask is not None:
        if damage_mask.shape[:2] != bgr.shape[:2] or damage_mask.ndim != 2:
            raise ValueError("Mask hư hại phải là ảnh xám cùng kích thước với ảnh đầu vào.")
        mask = cv2.bitwise_or(mask, np.uint8(damage_mask > 0) * 255)
    return cv2.inpaint(bgr, mask, 4, cv2.INPAINT_TELEA) if np.any(mask) else bgr.copy()


def improve_photo(bgr, *, light=True, contrast=True, denoise=True, sharpen=0.25,
                  neutralize=True, assessment=None):
    assessment = assessment or assess_image(bgr)
    result = bgr.copy()
    if denoise and assessment["impulse_fraction"] > 0.002:
        result = cv2.medianBlur(result, 3)
    elif denoise and assessment["noise"] >= 2:
        result = cv2.bilateralFilter(result, 5, 30, 30)
    if neutralize and assessment["sepia"]:
        result = cv2.cvtColor(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY),
                              cv2.COLOR_GRAY2BGR)
    if light and assessment["median_light"] < 95:
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        luminance = lab[:, :, 0].astype(np.float32) / 255.0
        lab[:, :, 0] = np.uint8(np.clip(np.power(luminance, 0.82) * 255, 0, 255))
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if contrast and assessment["light_range"] < 120:
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
        boosted = clahe.apply(lab[:, :, 0])
        lab[:, :, 0] = cv2.addWeighted(lab[:, :, 0], 0.45, boosted, 0.55, 0)
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    amount = float(np.clip(sharpen, 0, 1))
    if assessment["noise"] >= 3:
        amount *= 0.5
    if amount:
        blurred = cv2.GaussianBlur(result, (0, 0), 1.0)
        result = cv2.addWeighted(result, 1 + amount, blurred, -amount, 0)
    return result


def model_available():
    return MODEL_PATH.is_file() and MODEL_PATH.stat().st_size == MODEL_BYTES


def download_model(progress=None):
    """Download the published ONNX model once, checking size and SHA-256."""
    if model_available():
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial = MODEL_PATH.with_suffix(".onnx.part")
    digest = hashlib.sha256()
    total = 0
    try:
        with urlopen(MODEL_URL, timeout=60) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                total += len(chunk)
                if progress:
                    progress(total, MODEL_BYTES)
        if total != MODEL_BYTES or digest.hexdigest() != MODEL_SHA256:
            raise ValueError("Mô hình DDColor tải về không khớp SHA-256.")
        partial.replace(MODEL_PATH)
    finally:
        partial.unlink(missing_ok=True)
    return MODEL_PATH


@lru_cache(maxsize=1)
def _session():
    if not model_available():
        try:
            download_model()
        except Exception as exc:
            raise FileNotFoundError(
                "Thiếu mô hình DDColor. Hãy chạy 'python photo_fix.py --download-model' "
                "khi có Internet rồi thử lại."
            ) from exc
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(MODEL_PATH), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    input_shape = session.get_inputs()[0].shape
    output_shape = session.get_outputs()[0].shape
    if len(input_shape) != 4 or input_shape[1] != 3 or not all(
            isinstance(v, int) for v in input_shape[2:]):
        raise ValueError(f"Tensor đầu vào DDColor không đúng định dạng: {input_shape}")
    if len(output_shape) != 4 or output_shape[1] != 2:
        raise ValueError(f"Tensor đầu ra DDColor không đúng định dạng: {output_shape}")
    return session


def colorize_photo(bgr, saturation=1.0):
    """DDColor predicts Lab a/b from grayscale; original luminance stays intact."""
    session = _session()
    input_shape = session.get_inputs()[0].shape
    height, width = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    neutral = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    orig_l = cv2.cvtColor(neutral, cv2.COLOR_BGR2LAB)[:, :, :1]
    input_h, input_w = input_shape[2], input_shape[3]
    fit = min(input_w / width, input_h / height)
    small_w, small_h = max(1, round(width * fit)), max(1, round(height * fit))
    resized = cv2.resize(neutral, (small_w, small_h),
                         interpolation=cv2.INTER_AREA if fit < 1 else cv2.INTER_CUBIC)
    top, left = (input_h - small_h) // 2, (input_w - small_w) // 2
    small = cv2.copyMakeBorder(resized, top, input_h - small_h - top,
                               left, input_w - small_w - left,
                               cv2.BORDER_REFLECT_101)
    small_l = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[:, :, :1]
    gray_lab = np.dstack((small_l, np.zeros_like(small_l), np.zeros_like(small_l)))
    rgb = cv2.cvtColor(gray_lab, cv2.COLOR_LAB2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)
    predicted = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    ab = predicted[0].transpose(1, 2, 0)
    if ab.shape[:2] != (input_h, input_w):
        raise ValueError(f"Tensor màu DDColor sai kích thước: {ab.shape}")
    ab = ab[top:top + small_h, left:left + small_w]
    ab = cv2.resize(ab, (width, height), interpolation=cv2.INTER_LINEAR)
    ab *= float(np.clip(saturation, 0, 2))
    lab = np.concatenate((orig_l, ab), axis=2).astype(np.float32)
    colored = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return np.uint8(np.clip(colored * 255, 0, 255))


def run_photo_fix(image, *, mode=2, repair=True, light=True, contrast=True,
                  denoise=True, sharpen=0.25, saturation=1.0, neutralize=True,
                  damage_mask=None):
    """mode: 0 restore, 1 colorize, 2 restore then colorize grayscale only."""
    started = time.perf_counter()
    source = validate_image(image)
    mode = int(mode)
    if mode not in (0, 1, 2):
        raise ValueError("Chế độ phải là 0 (phục hồi), 1 (tô màu) hoặc 2 (kết hợp).")
    assessment = assess_image(source)
    stages = {"original": source.copy()}
    working = source.copy()
    if mode in (0, 2):
        if repair:
            working = repair_damage(working, damage_mask=damage_mask,
                                    sepia=assessment["sepia"])
        stages["repaired"] = working.copy()
        working = improve_photo(working, light=bool(light), contrast=bool(contrast),
                                denoise=bool(denoise), sharpen=sharpen,
                                neutralize=bool(neutralize),
                                assessment=assessment)
        stages["enhanced"] = working.copy()
    recolor = mode == 1 or (mode == 2 and
                            (assessment["grayscale"] or assessment["sepia"]))
    if recolor:
        working = colorize_photo(working, saturation=saturation)
        stages["colorized"] = working.copy()
    stages["final"] = working.copy()
    return PhotoFixResult(working, stages, assessment,
                          time.perf_counter() - started, recolor)


def save_stages(result, output_dir=ROOT / "outputs"):
    """Save each real pipeline stage without touching the source image."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = datetime.now().strftime("photofix_%Y%m%d_%H%M%S_%f")
    saved = {}
    for name, image in result.stages.items():
        path = output_dir / f"{prefix}_{name}.png"
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise OSError(f"Không mã hóa được ảnh ở bước {name}.")
        encoded.tofile(str(path))
        saved[name] = path
    report_path = output_dir / f"{prefix}_report.json"
    report_path.write_text(json.dumps({
        "assessment": result.assessment,
        "elapsed_seconds": result.elapsed_seconds,
        "colorized": result.colorized,
        "stages": {name: str(path.name) for name, path in saved.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return saved


def _main():
    import argparse

    parser = argparse.ArgumentParser(description="PhotoFix local: phục hồi và tô màu ảnh")
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--mask", type=Path, help="Mask trắng chỉ các vùng cần inpaint")
    parser.add_argument("--mode", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    if args.download_model:
        print(download_model())
        if args.input is None:
            return
    if args.input is None:
        parser.error("Cần --input hoặc --download-model.")
    source = cv2.imdecode(np.fromfile(str(args.input), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if source is None:
        parser.error("Không giải mã được ảnh đầu vào.")
    mask = None
    if args.mask:
        mask = cv2.imdecode(np.fromfile(str(args.mask), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            parser.error("Không giải mã được mask hư hại.")
    result = run_photo_fix(source, mode=args.mode, damage_mask=mask)
    saved = save_stages(result, args.output_dir)
    print(f"Hoàn tất trong {result.elapsed_seconds:.1f}s. Ảnh cuối: {saved['final']}")


if __name__ == "__main__":
    _main()

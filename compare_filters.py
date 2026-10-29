"""Create reproducible before/after boards for the course's spatial filters.

Usage: python compare_filters.py --input "path/to/photo.jpg" --output-dir demos
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from main import IP


def read_photo(path):
    data = np.fromfile(str(path), np.uint8)
    photo = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if photo is None:
        raise ValueError(f"Không đọc được ảnh: {path}")
    return photo


def make_test_inputs(clean, seed=42):
    """Add known noise and blur, so results can be compared to a clean reference."""
    rng = np.random.default_rng(seed)
    gaussian_noise = np.clip(clean.astype(np.float32) +
                             rng.normal(0, 20, clean.shape), 0, 255).astype(np.uint8)
    salt_pepper = clean.copy()
    mask = rng.random(clean.shape[:2])
    salt_pepper[mask < 0.015] = 0
    salt_pepper[(mask >= 0.015) & (mask < 0.03)] = 255
    blurred = cv2.GaussianBlur(clean, (5, 5), 1.2)
    return gaussian_noise, salt_pepper, blurred


def font(size):
    for path in (Path("C:/Windows/Fonts/arial.ttf"),
                 Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def draw_board(cards, title, note, path, size=360, columns=3):
    rows = (len(cards) + columns - 1) // columns
    gap, heading, footer = 16, 72, 46
    card_w, card_h = size, size + 74
    width = gap + columns * (card_w + gap)
    height = heading + rows * (card_h + gap) + footer
    board = Image.new("RGB", (width, height), "#111827")
    draw = ImageDraw.Draw(board)
    draw.text((gap, 13), title, font=font(27), fill="#f8fafc")
    for index, (label, image, detail) in enumerate(cards):
        x = gap + (index % columns) * (card_w + gap)
        y = heading + (index // columns) * (card_h + gap)
        draw.rounded_rectangle((x, y, x + card_w, y + card_h), radius=10,
                               fill="#1f2937")
        draw.text((x + 10, y + 7), label, font=font(18), fill="#f8fafc")
        draw.text((x + 10, y + 36), detail, font=font(13), fill="#93c5fd")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        picture = Image.fromarray(rgb)
        picture.thumbnail((size, size), Image.Resampling.LANCZOS)
        board.paste(picture, (x + (size - picture.width) // 2,
                              y + 68 + (size - picture.height) // 2))
    draw.text((gap, height - 34), note, font=font(13), fill="#cbd5e1")
    board.save(path)


def create_comparisons(input_path, output_dir):
    clean = read_photo(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gaussian_noise, salt_pepper, blurred = make_test_inputs(clean)
    boards = []
    for noise_name, noisy, file_name in (
        ("nhiễu Gaussian", gaussian_noise, "so_sanh_nhieu_gaussian.png"),
        ("nhiễu muối tiêu", salt_pepper, "so_sanh_nhieu_muoi_tieu.png"),
    ):
        denoised = [
            ("Trung bình", IP.mean_filter(noisy, k=3), "k=3"),
            ("Gaussian", IP.gaussian_filter(noisy, k=5, sigma=1.2), "k=5, σ=1.2"),
            ("Trung vị", IP.median_filter(noisy, k=3), "k=3"),
            ("Bilateral", IP.bilateral(noisy, d=9, sigma=50), "d=9, σ=50"),
        ]
        cards = [("Ảnh sạch tham chiếu", clean, "để đo chất lượng"),
                 ("Trước: ảnh có nhiễu", noisy, f"PSNR {cv2.PSNR(clean, noisy):.1f} dB")]
        for name, result, settings in denoised:
            cards.append(("Sau: " + name, result,
                          f"{settings} · PSNR {cv2.PSNR(clean, result):.1f} dB"))
        board_path = output_dir / file_name
        draw_board(cards, "Bộ lọc trên ảnh có " + noise_name,
                   "PSNR cao hơn nghĩa là ảnh gần ảnh sạch tham chiếu hơn.", board_path)
        boards.append(board_path)

    sharpened = IP.sharpen(blurred, strength=0.35)
    sharpen_cards = [("Ảnh sạch tham chiếu", clean, "để đo chất lượng"),
                     ("Trước: ảnh bị mờ", blurred,
                      f"Gaussian σ=1.2 · PSNR {cv2.PSNR(clean, blurred):.1f} dB"),
                     ("Sau: làm sắc nét", sharpened,
                      f"độ mạnh 0.35 · PSNR {cv2.PSNR(clean, sharpened):.1f} dB")]
    sharpen_path = output_dir / "so_sanh_lam_net.png"
    draw_board(sharpen_cards, "Làm sắc nét: so với ảnh bị mờ",
               "Quan sát tóc, mắt và đường biên; tăng nét quá mạnh có thể làm nổi nhiễu.",
               sharpen_path)
    return (*boards, sharpen_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tạo ảnh trước/sau cho các bộ lọc trong app")
    parser.add_argument("--input", required=True, type=Path, help="Ảnh gốc rõ nét")
    parser.add_argument("--output-dir", type=Path, default=Path("filter_comparisons"))
    args = parser.parse_args()
    for artifact in create_comparisons(args.input, args.output_dir):
        print(artifact)

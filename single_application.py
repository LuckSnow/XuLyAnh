"""Registration point for the project's one practical application: PhotoFix."""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from photo_fix import run_photo_fix


@dataclass(frozen=True)
class ApplicationSpec:
    name: str
    description: str
    process: Callable[..., np.ndarray]
    params: tuple[dict, ...] = field(default_factory=tuple)


def process_photo(img, **settings):
    return run_photo_fix(img, **settings).final


APPLICATION = ApplicationSpec(
    name="PhotoFix — Phục Hồi & Tô Màu",
    description=("Chế độ 0: phục hồi bằng OpenCV; 1: chỉ tô màu DDColor; "
                 "2: phục hồi rồi tự tô màu nếu ảnh gốc là ảnh xám hoặc sepia. "
                 "Màu do AI suy đoán, vết rách lớn cần mask riêng."),
    process=process_photo,
    params=(
        {"key": "mode", "label": "Chế Độ (0/1/2)", "min": 0, "max": 2,
         "step": 1, "default": 2, "is_int": True},
        {"key": "repair", "label": "Vá Đốm Sáng (0/1)", "min": 0, "max": 1,
         "step": 1, "default": 1, "is_int": True},
        {"key": "light", "label": "Sửa Ảnh Tối (0/1)", "min": 0, "max": 1,
         "step": 1, "default": 1, "is_int": True},
        {"key": "contrast", "label": "CLAHE (0/1)", "min": 0, "max": 1,
         "step": 1, "default": 1, "is_int": True},
        {"key": "denoise", "label": "Giảm Nhiễu (0/1)", "min": 0, "max": 1,
         "step": 1, "default": 1, "is_int": True},
        {"key": "neutralize", "label": "Khử Ám Nâu (0/1)", "min": 0, "max": 1,
         "step": 1, "default": 1, "is_int": True},
        {"key": "sharpen", "label": "Làm Sắc Nét", "min": 0, "max": 1,
         "step": 0.05, "default": 0.25, "is_int": False},
        {"key": "saturation", "label": "Độ Đậm Màu AI", "min": 0.5, "max": 1.5,
         "step": 0.1, "default": 1.0, "is_int": False},
    ),
)

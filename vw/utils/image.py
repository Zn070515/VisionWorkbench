"""图像工具函数."""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image


def load_image(path: str | Path) -> np.ndarray:
    """加载图片为 BGR numpy 数组."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"无法读取图片: {path}")
    return img


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    """BGR numpy 数组转为 PIL Image（RGB）."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_bytes(image: Image.Image, format: str = "JPEG") -> bytes:
    """PIL Image 转为字节."""
    import io
    buf = io.BytesIO()
    image.save(buf, format=format)
    return buf.getvalue()


def resize_to_max(image: np.ndarray, max_dim: int = 1024) -> np.ndarray:
    """等比缩放图片，确保最大边不超过 max_dim."""
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h))
    return image

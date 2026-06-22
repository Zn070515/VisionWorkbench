"""YOLO 数据集校验."""

from pathlib import Path
from typing import Optional


class ValidationResult:
    """校验结果."""

    def __init__(self):
        self.valid: bool = True
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []
        self.stats: dict = {}

    def add_error(self, msg: str):
        self.valid = False
        self.errors.append(msg)

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def add_info(self, msg: str):
        self.info.append(msg)

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
            "stats": self.stats,
        }


def validate_dataset(dataset_path: str | Path) -> ValidationResult:
    """校验 YOLO 格式数据集."""
    result = ValidationResult()
    root = Path(dataset_path).resolve()

    if not root.exists():
        result.add_error(f"路径不存在: {root}")
        return result

    _check_structure(root, result)
    _check_labels(root, result)
    _check_images(root, result)
    _check_classes(root, result)

    return result


def _check_structure(root: Path, result: ValidationResult):
    """检查 YOLO 目录结构."""
    # 需要 images/ 和 labels/ 目录 (或 train/val/test 子目录)
    has_images = (root / "images").is_dir()
    has_labels = (root / "labels").is_dir()
    has_yaml = list(root.glob("*.yaml")) or list(root.glob("*.yml"))

    if has_images and has_labels:
        result.add_info("✓ 标准 YOLO 结构: images/ + labels/")
        result.stats["structure"] = "standard"
    elif has_yaml:
        result.add_info("✓ 检测到 YAML 配置文件")
        result.stats["structure"] = "yaml_configured"
    else:
        result.add_warning("⚠ 非标准 YOLO 结构，建议包含 images/ 和 labels/ 目录")

    if has_yaml:
        result.add_info(f"  配置文件: {has_yaml[0].name}")
    else:
        result.add_warning("⚠ 未找到 .yaml 配置文件")


def _check_labels(root: Path, result: ValidationResult):
    """检查标注文件."""
    label_dirs = _find_label_dirs(root)
    if not label_dirs:
        result.add_error("未找到 labels 目录")
        return

    total = 0
    valid_count = 0
    broken_files = []

    for label_dir in label_dirs:
        for label_file in sorted(label_dir.glob("*.txt")):
            total += 1
            try:
                with open(label_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) < 5:
                            broken_files.append(str(label_file.name))
                            break
                        # 验证数值格式
                        for p in parts:
                            float(p)
                valid_count += 1
            except Exception:
                broken_files.append(str(label_file.name))

    result.add_info(f"✓ 标注文件: {valid_count}/{total} 有效")
    result.stats["total_labels"] = total
    result.stats["valid_labels"] = valid_count
    result.stats["broken_labels"] = len(broken_files)

    if broken_files:
        result.add_error(f"✗ {len(broken_files)} 个标注文件损坏: {', '.join(broken_files[:5])}")
        result.stats["broken_label_files"] = broken_files


def _check_images(root: Path, result: ValidationResult):
    """检查图片文件."""
    import cv2

    image_dirs = _find_image_dirs(root)
    if not image_dirs:
        result.add_error("未找到 images 目录")
        return

    total = 0
    valid_count = 0
    broken = []
    small_images = []

    for img_dir in image_dirs:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            for img_file in img_dir.glob(ext):
                total += 1
                try:
                    img = cv2.imread(str(img_file))
                    if img is None:
                        broken.append(str(img_file.name))
                        continue
                    h, w = img.shape[:2]
                    if w < 32 or h < 32:
                        small_images.append(f"{img_file.name}({w}x{h})")
                    valid_count += 1
                except Exception:
                    broken.append(str(img_file.name))

    result.add_info(f"✓ 图片文件: {valid_count}/{total} 可读")
    result.stats["total_images"] = total
    result.stats["valid_images"] = valid_count
    result.stats["broken_images"] = len(broken)

    if broken:
        result.add_error(f"✗ {len(broken)} 张图片损坏: {', '.join(broken[:5])}")
        result.stats["broken_image_files"] = broken

    if small_images:
        result.add_warning(f"⚠ {len(small_images)} 张图片尺寸过小(<32px): {', '.join(small_images[:5])}")
        result.stats["small_images"] = small_images


def _check_classes(root: Path, result: ValidationResult):
    """检查类别统计."""
    label_dirs = _find_label_dirs(root)
    class_ids = set()
    class_sample_count = {}

    for label_dir in label_dirs:
        for label_file in label_dir.glob("*.txt"):
            try:
                with open(label_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            class_ids.add(cls_id)
                            class_sample_count[cls_id] = class_sample_count.get(cls_id, 0) + 1
            except Exception:
                pass

    result.stats["num_classes"] = len(class_ids)
    result.stats["class_distribution"] = class_sample_count

    if class_ids:
        result.add_info(f"✓ 类别数: {len(class_ids)}")
        for cid, count in sorted(class_sample_count.items()):
            result.add_info(f"  类别 {cid}: {count} 个标注框")
    else:
        result.add_warning("⚠ 未检测到任何标注类别")


def _find_label_dirs(root: Path) -> list[Path]:
    """查找所有 labels 目录."""
    dirs = []
    if (root / "labels").is_dir():
        dirs.append(root / "labels")
    for sub in ("train", "val", "test"):
        p = root / sub / "labels"
        if p.is_dir():
            dirs.append(p)
        p = root / "labels" / sub
        if p.is_dir():
            dirs.append(p)
    return dirs or [root]  # fallback to root


def _find_image_dirs(root: Path) -> list[Path]:
    """查找所有 images 目录."""
    dirs = []
    if (root / "images").is_dir():
        dirs.append(root / "images")
    for sub in ("train", "val", "test"):
        p = root / sub / "images"
        if p.is_dir():
            dirs.append(p)
        p = root / "images" / sub
        if p.is_dir():
            dirs.append(p)
    return dirs or [root]

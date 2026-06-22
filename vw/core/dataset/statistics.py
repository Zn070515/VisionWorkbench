"""数据集统计."""

from pathlib import Path


def get_statistics(dataset_path: str | Path) -> dict:
    """获取数据集统计信息."""
    root = Path(dataset_path).resolve()
    from vw.core.dataset.validator import validate_dataset, _find_image_dirs, _find_label_dirs

    result = validate_dataset(root)
    stats = result.stats.copy()

    # 图片尺寸分布
    import cv2
    widths = []
    heights = []
    image_dirs = _find_image_dirs(root)
    for img_dir in image_dirs:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            for img_file in img_dir.glob(ext):
                try:
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        h, w = img.shape[:2]
                        widths.append(w)
                        heights.append(h)
                except Exception:
                    pass

    if widths:
        stats["image_size"] = {
            "avg_width": round(sum(widths) / len(widths)),
            "avg_height": round(sum(heights) / len(heights)),
            "min_width": min(widths),
            "min_height": min(heights),
            "max_width": max(widths),
            "max_height": max(heights),
        }
        stats["total_images_sampled"] = len(widths)

    # 标注框尺寸分布
    label_dirs = _find_label_dirs(root)
    box_widths = []
    box_heights = []
    for label_dir in label_dirs:
        for label_file in label_dir.glob("*.txt"):
            try:
                with open(label_file, encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            bw = float(parts[3])
                            bh = float(parts[4])
                            box_widths.append(bw)
                            box_heights.append(bh)
            except Exception:
                pass

    if box_widths:
        stats["bbox_stats"] = {
            "avg_width": round(sum(box_widths) / len(box_widths), 4),
            "avg_height": round(sum(box_heights) / len(box_heights), 4),
            "min_width": round(min(box_widths), 4),
            "max_width": round(max(box_widths), 4),
        }
        # 小/中/大目标分布
        tiny = sum(1 for w in box_widths if w < 0.05)
        small = sum(1 for w in box_widths if 0.05 <= w < 0.2)
        medium = sum(1 for w in box_widths if 0.2 <= w < 0.5)
        large = sum(1 for w in box_widths if w >= 0.5)
        stats["bbox_size_distribution"] = {
            "tiny (<5%)": tiny,
            "small (5-20%)": small,
            "medium (20-50%)": medium,
            "large (>50%)": large,
        }

    return stats

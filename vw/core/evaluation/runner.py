"""评估运行器 — 批量推理、视频评估、错误分析."""

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from vw.core.model.inference import InferenceEngine
from vw.infrastructure.database import get_db


def _box_iou(box_a: tuple, box_b: tuple) -> float:
    """计算两个边界框的 IoU。(x1, y1, x2, y2) 格式。"""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _normalized_to_pixel(cx, cy, w, h, img_w, img_h) -> tuple:
    """YOLO 归一化坐标 → 像素 (x1, y1, x2, y2)。"""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return (int(x1), int(y1), int(x2), int(y2))


class EvaluationRunner:
    """批量推理与错误分析."""

    def __init__(self):
        self.engine = InferenceEngine()

    # ── 数据集批量评估 ────────────────────────────────────

    def run_on_dataset(
        self,
        model_path: str | Path,
        dataset_path: str | Path,
        confidence: float = 0.5,
        iou_threshold: float = 0.5,
        device: str = "cpu",
        max_images: Optional[int] = None,
    ) -> dict:
        """在数据集上运行评估，返回完整报告。"""
        if not self.engine.is_loaded or self.engine.model_path != str(model_path):
            self.engine.load(model_path)

        images_dir = Path(dataset_path) / "images"
        labels_dir = Path(dataset_path) / "labels"

        if not images_dir.exists():
            return {"error": "images 目录不存在", "valid": False}

        image_files = sorted(
            p for p in images_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if max_images:
            image_files = image_files[:max_images]

        results = {
            "total_images": len(image_files),
            "total_gt_boxes": 0,
            "total_pred_boxes": 0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "per_class": {},
            "per_image": [],
        }

        for img_path in image_files:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            h, w = image.shape[:2]

            # 读取真值
            gt_boxes = self._read_labels(labels_dir / f"{img_path.stem}.txt", w, h)

            # 推理
            detections = self.engine.predict(image, confidence=confidence, device=device)
            pred_boxes = [
                (d["class_id"], d["x1"], d["y1"], d["x2"], d["y2"], d["confidence"])
                for d in detections
            ]

            # 匹配
            img_result = self._match_boxes(gt_boxes, pred_boxes, iou_threshold)

            results["total_gt_boxes"] += len(gt_boxes)
            results["total_pred_boxes"] += len(pred_boxes)
            results["true_positives"] += img_result["tp"]
            results["false_positives"] += img_result["fp"]
            results["false_negatives"] += img_result["fn"]

            for cls_id in img_result["per_class"]:
                if cls_id not in results["per_class"]:
                    results["per_class"][cls_id] = {
                        "tp": 0, "fp": 0, "fn": 0,
                        "class_name": img_result["per_class"][cls_id].get("class_name", str(cls_id)),
                    }
                for k in ("tp", "fp", "fn"):
                    results["per_class"][cls_id][k] += img_result["per_class"][cls_id].get(k, 0)

            results["per_image"].append(img_result)

        # 计算每类指标
        for cls_id, counts in results["per_class"].items():
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            counts["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            counts["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            counts["f1"] = (
                2 * counts["precision"] * counts["recall"]
                / (counts["precision"] + counts["recall"])
                if (counts["precision"] + counts["recall"]) > 0
                else 0.0
            )

        # 汇总
        tp = results["true_positives"]
        fp = results["false_positives"]
        fn = results["false_negatives"]
        results["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        results["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        results["mAP"] = results["true_positives"] / max(results["total_gt_boxes"], 1)
        results["valid"] = True

        return results

    # ── 视频评估 ──────────────────────────────────────────

    def run_on_video(
        self,
        model_path: str | Path,
        video_path: str | Path,
        confidence: float = 0.5,
        device: str = "cpu",
        sample_every: int = 1,
    ) -> dict:
        """对视频逐帧推理，返回帧级检测结果。"""
        if not self.engine.is_loaded or self.engine.model_path != str(model_path):
            self.engine.load(model_path)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"error": "无法打开视频", "valid": False}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_every == 0:
                detections = self.engine.predict(frame, confidence=confidence, device=device)
                frames.append({
                    "frame_idx": frame_idx,
                    "timestamp": frame_idx / fps,
                    "detection_count": len(detections),
                    "detections": detections,
                })
            frame_idx += 1

        cap.release()

        return {
            "total_frames": total_frames,
            "sampled_frames": len(frames),
            "fps": fps,
            "resolution": f"{width}x{height}",
            "total_detections": sum(f["detection_count"] for f in frames),
            "frames": frames,
            "valid": True,
        }

    # ── 错误分析 ──────────────────────────────────────────

    def error_analysis(self, eval_results: dict) -> dict:
        """从评估结果生成错误分析报告。"""
        per_class = eval_results.get("per_class", {})
        if not per_class:
            return {"worst_classes": [], "confidence_distribution": {}}

        # 按 F1 排序找最差类别
        sorted_classes = sorted(
            per_class.items(),
            key=lambda kv: kv[1].get("f1", 0),
        )

        worst = []
        for cls_id, counts in sorted_classes[:5]:
            if counts.get("f1", 1.0) < 1.0:
                worst.append({
                    "class_id": cls_id,
                    "class_name": counts.get("class_name", str(cls_id)),
                    "precision": counts.get("precision", 0),
                    "recall": counts.get("recall", 0),
                    "f1": counts.get("f1", 0),
                    "fp": counts.get("fp", 0),
                    "fn": counts.get("fn", 0),
                })

        # 置信度分布（来自 per_image 结果）
        conf_dist = {"high": 0, "medium": 0, "low": 0}
        for img_res in eval_results.get("per_image", []):
            for err in img_res.get("false_positives_detail", []):
                conf = err.get("confidence", 0)
                if conf >= 0.7:
                    conf_dist["high"] += 1
                elif conf >= 0.3:
                    conf_dist["medium"] += 1
                else:
                    conf_dist["low"] += 1

        return {
            "worst_classes": worst,
            "confidence_distribution": conf_dist,
        }

    # ── 持久化 ────────────────────────────────────────────

    def save(
        self,
        model_version_id: int,
        eval_type: str,
        input_path: str,
        results: dict,
        dataset_id: Optional[int] = None,
    ) -> int:
        """保存评估结果到数据库。"""
        db = get_db()
        # 不存储逐帧/逐图详情以控制行大小
        slim = {k: v for k, v in results.items()
                if k not in ("frames", "per_image")}
        row_id = db.insert("evaluation", {
            "model_version_id": model_version_id,
            "dataset_id": dataset_id,
            "type": eval_type,
            "input_path": input_path,
            "results": json.dumps(slim, ensure_ascii=False),
        })
        return row_id

    def get_history(self, limit: int = 50) -> list[dict]:
        db = get_db()
        rows = db.fetchall(
            "SELECT * FROM evaluation ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in rows]

    # ── 辅助方法 ──────────────────────────────────────────

    def _read_labels(self, label_path: Path, img_w: int, img_h: int) -> list:
        """读取 YOLO 标签文件，返回 [(class_id, x1, y1, x2, y2), ...]."""
        boxes = []
        if not label_path.exists():
            return boxes
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    pixel_box = _normalized_to_pixel(cx, cy, w, h, img_w, img_h)
                    boxes.append((cls_id,) + pixel_box)
                except (ValueError, IndexError):
                    continue
        return boxes

    def _match_boxes(self, gt_boxes: list, pred_boxes: list, iou_threshold: float) -> dict:
        """匹配预测框与真值框，返回 TP/FP/FN。"""
        matched_gt = set()
        matched_pred = set()
        tp = 0
        fp_detail = []

        for pi, pred in enumerate(pred_boxes):
            p_cls, px1, py1, px2, py2, p_conf = pred
            best_iou = 0
            best_gi = -1
            for gi, gt in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                g_cls = gt[0]
                if g_cls != p_cls:
                    continue
                iou = _box_iou((px1, py1, px2, py2), (gt[1], gt[2], gt[3], gt[4]))
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_iou >= iou_threshold and best_gi >= 0:
                tp += 1
                matched_gt.add(best_gi)
                matched_pred.add(pi)
            else:
                fp_detail.append({
                    "class_id": p_cls,
                    "confidence": p_conf,
                    "box": [px1, py1, px2, py2],
                })

        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp

        per_class = {}
        for gi, gt in enumerate(gt_boxes):
            g_cls = gt[0]
            if g_cls not in per_class:
                per_class[g_cls] = {"tp": 0, "fp": 0, "fn": 0, "class_name": str(g_cls)}
        for pred in pred_boxes:
            p_cls = pred[0]
            if p_cls not in per_class:
                per_class[p_cls] = {"tp": 0, "fp": 0, "fn": 0, "class_name": str(p_cls)}

        for gi in matched_gt:
            g_cls = gt_boxes[gi][0]
            if g_cls in per_class:
                per_class[g_cls]["tp"] += 1
        for gi, gt in enumerate(gt_boxes):
            if gi not in matched_gt:
                g_cls = gt[0]
                if g_cls in per_class:
                    per_class[g_cls]["fn"] += 1
        for pi, pred in enumerate(pred_boxes):
            if pi not in matched_pred:
                p_cls = pred[0]
                if p_cls in per_class:
                    per_class[p_cls]["fp"] += 1

        return {
            "tp": tp, "fp": fp, "fn": fn,
            "per_class": per_class,
            "false_positives_detail": fp_detail,
        }

"""YOLO 推理引擎 — v0.1 核心."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO


class InferenceEngine:
    """加载 YOLO 模型并执行推理."""

    def __init__(self):
        self._model: Optional[YOLO] = None
        self._model_path: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_path(self) -> Optional[str]:
        return self._model_path

    @property
    def model_info(self) -> dict:
        if not self._model:
            return {}
        names = self._model.names or {}
        return {
            "path": self._model_path,
            "num_classes": len(names),
            "classes": names,
        }

    def load(self, model_path: str | Path) -> dict:
        """加载模型文件，返回模型信息。非 YOLO 格式抛出 ValueError。"""
        path = str(Path(model_path).resolve())
        try:
            self._model = YOLO(path)
        except (TypeError, RuntimeError, FileNotFoundError) as e:
            raise ValueError(
                f"无法加载模型文件。请确保上传的是 Ultralytics YOLO 官方权重 (.pt) 文件。\n"
                f"原始错误: {e}"
            ) from e
        self._model_path = path
        return self.model_info

    def predict(
        self,
        image: np.ndarray,
        confidence: float = 0.5,
        iou: float = 0.45,
        device: str = "cpu",
    ) -> list[dict]:
        """对单张图片执行推理，返回检测结果列表."""
        if not self._model:
            raise RuntimeError("模型未加载")

        results = self._model(
            image,
            conf=confidence,
            iou=iou,
            device=device,
            verbose=False,
        )

        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                cls_name = self._model.names.get(cls_id, str(cls_id))
                detections.append({
                    "x1": int(xyxy[0]),
                    "y1": int(xyxy[1]),
                    "x2": int(xyxy[2]),
                    "y2": int(xyxy[3]),
                    "confidence": round(conf, 4),
                    "class_id": cls_id,
                    "class_name": cls_name,
                })
        return detections

    def draw_detections(
        self,
        image: np.ndarray,
        detections: list[dict],
        show_conf: bool = True,
    ) -> np.ndarray:
        """在图片上绘制检测框和标签."""
        output = image.copy()
        for d in detections:
            x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{d['class_name']} {d['confidence']:.2f}" if show_conf else d["class_name"]
            cv2.putText(output, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return output

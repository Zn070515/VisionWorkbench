"""评估运行器单元测试."""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from vw.core.evaluation.runner import (
    EvaluationRunner, _box_iou, _normalized_to_pixel,
)
from vw.core.dataset.registry import DatasetRegistry
from vw.core.model.registry import ModelRegistry
from vw.infrastructure.database import get_db, reset_db


@pytest.fixture(autouse=True)
def _reset_db():
    reset_db()
    yield
    reset_db()


def create_mini_yolo_dataset(base: Path):
    """创建最小 YOLO 数据集用于评估测试."""
    images = base / "images"
    labels = base / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)

    for i in range(3):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.imwrite(str(images / f"img_{i}.jpg"), img)
        # 一个 GT 框
        with open(labels / f"img_{i}.txt", "w") as f:
            f.write("0 0.5 0.5 0.25 0.25\n")

    return base


class TestBoxIoU:
    def test_perfect_overlap(self):
        iou = _box_iou((0, 0, 100, 100), (0, 0, 100, 100))
        assert iou == 1.0

    def test_no_overlap(self):
        iou = _box_iou((0, 0, 10, 10), (90, 90, 100, 100))
        assert iou == 0.0

    def test_half_overlap(self):
        iou = _box_iou((0, 0, 100, 100), (50, 0, 150, 100))
        assert 0.3 < iou < 0.4


class TestCoordinateConversion:
    def test_center_box(self):
        result = _normalized_to_pixel(0.5, 0.5, 0.5, 0.5, 100, 100)
        assert result == (25, 25, 75, 75)


class TestLabelReading:
    def test_read_labels(self, tmp_path):
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        label_file = labels_dir / "test.txt"
        label_file.write_text("0 0.5 0.5 0.2 0.2\n2 0.8 0.8 0.1 0.1\n")
        runner = EvaluationRunner()
        boxes = runner._read_labels(label_file, 640, 480)
        assert len(boxes) == 2
        assert boxes[0][0] == 0  # class_id
        assert boxes[1][0] == 2

    def test_read_missing_labels(self, tmp_path):
        runner = EvaluationRunner()
        boxes = runner._read_labels(tmp_path / "nonexistent.txt", 640, 480)
        assert boxes == []


class TestBoxMatching:
    def test_perfect_match(self):
        runner = EvaluationRunner()
        gt = [(0, 0, 0, 100, 100)]
        pred = [(0, 0, 0, 100, 100, 0.9)]
        result = runner._match_boxes(gt, pred, 0.5)
        assert result["tp"] == 1
        assert result["fp"] == 0
        assert result["fn"] == 0

    def test_class_mismatch(self):
        runner = EvaluationRunner()
        gt = [(0, 0, 0, 100, 100)]
        pred = [(1, 0, 0, 100, 100, 0.9)]
        result = runner._match_boxes(gt, pred, 0.5)
        assert result["tp"] == 0
        assert result["fp"] == 1
        assert result["fn"] == 1

    def test_extra_prediction(self):
        runner = EvaluationRunner()
        gt = [(0, 0, 0, 100, 100)]
        pred = [
            (0, 0, 0, 100, 100, 0.9),  # 匹配
            (0, 200, 200, 300, 300, 0.5),  # 误检
        ]
        result = runner._match_boxes(gt, pred, 0.5)
        assert result["tp"] == 1
        assert result["fp"] == 1
        assert result["fn"] == 0

    def test_missed_gt(self):
        runner = EvaluationRunner()
        gt = [(0, 0, 0, 100, 100), (1, 200, 200, 300, 300)]
        pred = [(0, 0, 0, 100, 100, 0.9)]
        result = runner._match_boxes(gt, pred, 0.5)
        assert result["tp"] == 1
        assert result["fp"] == 0
        assert result["fn"] == 1


class TestSaveAndHistory:
    def test_save_and_list(self, tmp_path):
        import torch
        db_path = Path(tempfile.mkdtemp()) / "test_eval.db"
        get_db(db_path)
        # 创建模型和版本以满足外键约束
        reg = ModelRegistry(workspace=Path(tempfile.mkdtemp()) / "mw")
        dummy = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        torch.save({"dummy": True}, dummy.name)
        mid = reg.register("test_m", dummy.name)
        versions = reg.get_versions(mid)
        version_id = versions[0]["id"]

        runner = EvaluationRunner()
        rid = runner.save(
            model_version_id=version_id,
            eval_type="image",
            input_path="/tmp/test",
            results={"valid": True, "precision": 0.9},
            dataset_id=None,
        )
        assert rid > 0
        history = runner.get_history()
        assert len(history) == 1
        assert history[0]["type"] == "image"
        reset_db()

    def test_history_empty(self):
        db_path = Path(tempfile.mkdtemp()) / "test_eval2.db"
        get_db(db_path)
        runner = EvaluationRunner()
        history = runner.get_history()
        assert history == []
        reset_db()


class TestErrorAnalysis:
    def test_analysis_empty(self):
        runner = EvaluationRunner()
        analysis = runner.error_analysis({"per_class": {}})
        assert analysis["worst_classes"] == []

    def test_analysis_with_data(self):
        runner = EvaluationRunner()
        results = {
            "per_class": {
                "0": {"class_name": "cat", "tp": 10, "fp": 2, "fn": 3,
                      "precision": 0.833, "recall": 0.769, "f1": 0.8},
                "1": {"class_name": "dog", "tp": 5, "fp": 5, "fn": 5,
                      "precision": 0.5, "recall": 0.5, "f1": 0.5},
            },
            "per_image": [],
        }
        analysis = runner.error_analysis(results)
        worst = analysis["worst_classes"]
        assert len(worst) <= 5
        assert worst[0]["class_name"] == "dog"  # worse F1 first

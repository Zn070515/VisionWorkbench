"""InferenceEngine 单元测试."""

import numpy as np
import pytest

from vw.core.model.inference import InferenceEngine


def test_engine_initially_not_loaded():
    engine = InferenceEngine()
    assert not engine.is_loaded
    assert engine.model_path is None
    assert engine.model_info == {}


def test_predict_raises_when_not_loaded():
    engine = InferenceEngine()
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="模型未加载"):
        engine.predict(image)


def test_draw_detections():
    engine = InferenceEngine()
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    detections = [
        {"x1": 10, "y1": 20, "x2": 100, "y2": 150,
         "class_name": "cat", "confidence": 0.95,
         "class_id": 0},
    ]
    output = engine.draw_detections(image, detections)
    assert output.shape == image.shape
    # 应该有绿色框（非全黑）
    assert output.sum() > image.sum()

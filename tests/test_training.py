"""训练模块单元测试."""

import json
import tempfile
import time
from pathlib import Path

import pytest

from vw.core.training.manager import TrainingManager, _find_yolo
from vw.infrastructure.database import get_db, reset_db


@pytest.fixture(autouse=True)
def _reset_db():
    reset_db()
    yield
    reset_db()


@pytest.fixture
def manager():
    db_path = Path(tempfile.mkdtemp()) / "test_train.db"
    get_db(db_path)
    mgr = TrainingManager(
        workspace=Path(tempfile.mkdtemp()) / "train_workspace"
    )
    yield mgr
    reset_db()


def test_yolo_cli_found():
    """验证 yolo CLI 可被找到."""
    path = _find_yolo()
    assert path
    assert Path(path).exists() or "yolo" in path


class TestCreateAndList:
    def test_create_task(self, manager):
        tid = manager.create_task("test_train", "yolov8n.pt", {
            "data": "/tmp/fake.yaml",
            "epochs": 10,
            "imgsz": 320,
            "batch": 8,
            "device": "cpu",
        })
        assert tid > 0

        task = manager.get_task(tid)
        assert task["model_name"] == "test_train"
        assert task["base_model"] == "yolov8n.pt"
        assert task["status"] == "pending"
        config = json.loads(task["config"])
        assert config["epochs"] == 10

    def test_list_tasks(self, manager):
        manager.create_task("a", "yolov8n.pt", {})
        manager.create_task("b", "yolov8s.pt", {})
        tasks = manager.list_tasks()
        assert len(tasks) == 2

    def test_filter_by_status(self, manager):
        manager.create_task("a", "yolov8n.pt", {})
        manager.create_task("b", "yolov8s.pt", {})
        running = manager.list_tasks(status="running")
        assert running == []
        pending = manager.list_tasks(status="pending")
        assert len(pending) == 2


class TestDelete:
    def test_delete_pending(self, manager):
        tid = manager.create_task("x", "yolov8n.pt", {})
        manager.delete_task(tid)
        assert manager.get_task(tid) is None

    def test_delete_nonexistent_does_not_raise(self, manager):
        manager.delete_task(99999)


class TestLogAndMetrics:
    def test_get_log_empty(self, manager):
        tid = manager.create_task("x", "yolov8n.pt", {})
        log = manager.get_log(tid)
        assert log == ""

    def test_get_log_with_content(self, manager):
        tid = manager.create_task("x", "yolov8n.pt", {})
        task = manager.get_task(tid)
        log_path = Path(task["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("line 1\nline 2\nline 3\n")
        log = manager.get_log(tid, tail=2)
        assert "line 2" in log
        assert "line 3" in log
        assert "line 1" not in log

    def test_metrics_parsing(self, manager):
        tid = manager.create_task("x", "yolov8n.pt", {})
        task = manager.get_task(tid)
        log_path = Path(task["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
            "   1/100      1.23G      1.456      2.345      1.234        123        640\n"
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)\n"
            "                   all        100        500      0.789      0.723      0.782      0.567\n"
        )
        metrics = manager.get_metrics(tid)
        assert len(metrics) == 1
        m = metrics[0]
        assert m["epoch"] == 1
        assert m["box_loss"] == 1.456
        assert m["cls_loss"] == 2.345
        assert abs(m["map50"] - 0.782) < 0.001


class TestStatusTracking:
    def test_poll_completion_pending(self, manager):
        tid = manager.create_task("x", "yolov8n.pt", {})
        result = manager.poll_completion(tid)
        assert result is None

    def test_stop_pending_noop(self, manager):
        tid = manager.create_task("x", "yolov8n.pt", {})
        stopped = manager.stop(tid)
        assert stopped is False
        task = manager.get_task(tid)
        assert task["status"] == "pending"


class TestBuildCommand:
    def test_build_command(self, manager):
        tid = manager.create_task("my_model", "yolov8s.pt", {
            "data": "/data/data.yaml",
            "epochs": 50,
            "imgsz": 640,
            "batch": 16,
            "device": "0",
        })
        task = manager.get_task(tid)
        config = json.loads(task["config"])
        cmd = manager._build_command(task, config)
        assert "yolo" in cmd[0] or cmd[0].endswith("yolo.exe")
        assert "train" == cmd[1]
        cmd_str = " ".join(cmd)
        assert "model=yolov8s.pt" in cmd_str
        assert "data=/data/data.yaml" in cmd_str
        assert "epochs=50" in cmd_str
        assert "device=0" in cmd_str

"""模型注册表单元测试."""

import tempfile
from pathlib import Path

import pytest

from vw.core.model.registry import ModelRegistry
from vw.core.training.manager import TrainingManager
from vw.infrastructure.database import get_db, reset_db


@pytest.fixture(autouse=True)
def _reset_db():
    reset_db()
    yield
    reset_db()


@pytest.fixture
def registry():
    db_path = Path(tempfile.mkdtemp()) / "test_model.db"
    get_db(db_path)
    reg = ModelRegistry(workspace=Path(tempfile.mkdtemp()) / "model_workspace")
    yield reg
    reset_db()


@pytest.fixture
def dummy_pt():
    """创建一个最小的 dummy .pt 文件."""
    import torch
    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    torch.save({"dummy": True}, tmp.name)
    return tmp.name


class TestRegisterAndGet:
    def test_register(self, registry, dummy_pt):
        mid = registry.register("test_model", dummy_pt, task_type="detect",
                                num_classes=80, input_size=640)
        assert mid > 0
        m = registry.get(mid)
        assert m["name"] == "test_model"
        assert m["task_type"] == "detect"

    def test_register_creates_v1(self, registry, dummy_pt):
        mid = registry.register("m", dummy_pt)
        versions = registry.get_versions(mid)
        assert len(versions) == 1
        assert versions[0]["version"] == 1

    def test_get_by_name(self, registry, dummy_pt):
        registry.register("unique_model", dummy_pt)
        m = registry.get_by_name("unique_model")
        assert m is not None
        assert m["name"] == "unique_model"

    def test_get_by_name_missing(self, registry):
        m = registry.get_by_name("nope")
        assert m is None

    def test_list_all(self, registry, dummy_pt):
        registry.register("a", dummy_pt)
        registry.register("b", dummy_pt)
        assert len(registry.list_all()) == 2

    def test_delete(self, registry, dummy_pt):
        mid = registry.register("to_delete", dummy_pt)
        registry.delete(mid)
        assert registry.get(mid) is None
        assert registry.get_versions(mid) == []


class TestVersioning:
    def test_add_version(self, registry, dummy_pt):
        mid = registry.register("m", dummy_pt)
        vid = registry.add_version(mid, dummy_pt, notes="improved",
                                   metrics={"map50": 0.85, "map50_95": 0.62})
        assert vid > 0
        versions = registry.get_versions(mid)
        assert len(versions) == 2
        assert versions[0]["version"] == 2
        assert versions[0]["notes"] == "improved"
        assert versions[0]["map50"] == 0.85

    def test_get_latest_version(self, registry, dummy_pt):
        mid = registry.register("m", dummy_pt)
        registry.add_version(mid, dummy_pt)
        latest = registry.get_latest_version(mid)
        assert latest["version"] == 2


class TestDefault:
    def test_set_default(self, registry, dummy_pt):
        mid1 = registry.register("a", dummy_pt)
        mid2 = registry.register("b", dummy_pt)
        assert registry.get_default() is None
        registry.set_default(mid1)
        d = registry.get_default()
        assert d["id"] == mid1
        registry.set_default(mid2)
        d = registry.get_default()
        assert d["id"] == mid2

    def test_only_one_default(self, registry, dummy_pt):
        mid1 = registry.register("a", dummy_pt)
        mid2 = registry.register("b", dummy_pt)
        registry.set_default(mid1)
        registry.set_default(mid2)
        default = registry.get_default()
        assert default["id"] == mid2
        a = registry.get(mid1)
        assert a["is_default"] == 0


class TestLineage:
    def test_empty_lineage(self, registry, dummy_pt):
        mid = registry.register("orphan", dummy_pt)
        lineage = registry.get_lineage(mid)
        assert lineage["model"]["name"] == "orphan"
        assert len(lineage["versions"]) == 1

    def test_lineage_with_training(self, registry, dummy_pt):
        # 创建训练任务
        mgr = TrainingManager(
            workspace=Path(tempfile.mkdtemp()) / "train_ws"
        )
        tid = mgr.create_task("train_x", "yolov8n.pt", {
            "data": "/tmp/data.yaml", "epochs": 10,
        })
        # 注册模型并关联训练
        mid = registry.register("m", dummy_pt)
        registry.add_version(mid, dummy_pt, training_task_id=tid,
                             dataset_id=None, notes="v2 from training")
        lineage = registry.get_lineage(mid)
        assert len(lineage["versions"]) == 2
        v2 = lineage["versions"][0]
        assert v2["training_task"] is not None
        assert v2["training_task"]["model_name"] == "train_x"


class TestRegisterFromTraining:
    def test_new_model_from_training(self, registry, dummy_pt):
        mgr = TrainingManager(
            workspace=Path(tempfile.mkdtemp()) / "train_ws2"
        )
        tid = mgr.create_task("trained_model", "yolov8n.pt", {"epochs": 100})
        mid = registry.register_from_training(tid, dummy_pt,
                                              metrics={"map50": 0.9})
        assert mid > 0
        m = registry.get(mid)
        assert m["name"] == "trained_model"

    def test_append_version_from_training(self, registry, dummy_pt):
        mgr = TrainingManager(
            workspace=Path(tempfile.mkdtemp()) / "train_ws3"
        )
        tid1 = mgr.create_task("recurring", "yolov8n.pt", {"epochs": 10})
        tid2 = mgr.create_task("recurring", "yolov8n.pt", {"epochs": 50})
        mid = registry.register_from_training(tid1, dummy_pt)
        assert len(registry.get_versions(mid)) == 1
        registry.register_from_training(tid2, dummy_pt,
                                        metrics={"map50": 0.88})
        assert len(registry.get_versions(mid)) == 2


class TestExport:
    def test_export(self, registry, dummy_pt):
        mid = registry.register("export_me", dummy_pt)
        dest = Path(tempfile.mkdtemp()) / "exported"
        registry.export(mid, dest)
        files = list(dest.glob("*.pt"))
        assert len(files) == 1

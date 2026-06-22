"""实验管理单元测试."""

import tempfile
from pathlib import Path

import pytest

from vw.core.experiment.manager import ExperimentManager
from vw.core.training.manager import TrainingManager
from vw.infrastructure.database import get_db, reset_db


@pytest.fixture(autouse=True)
def _reset_db():
    reset_db()
    yield
    reset_db()


@pytest.fixture
def em():
    db_path = Path(tempfile.mkdtemp()) / "test_exp.db"
    get_db(db_path)
    mgr = ExperimentManager()
    yield mgr
    reset_db()


class TestCreateAndGet:
    def test_create(self, em):
        eid = em.create("exp1", description="test desc",
                        hypothesis="it works", tags=["a", "b"])
        assert eid > 0
        exp = em.get(eid)
        assert exp["name"] == "exp1"
        assert exp["description"] == "test desc"
        assert exp["hypothesis"] == "it works"
        assert exp["tags"] == ["a", "b"]

    def test_list_all(self, em):
        em.create("a")
        em.create("b")
        assert len(em.list_all()) == 2

    def test_update(self, em):
        eid = em.create("x")
        em.update(eid, conclusion="it worked", tags=["c"])
        exp = em.get(eid)
        assert exp["conclusion"] == "it worked"
        assert exp["tags"] == ["c"]

    def test_delete(self, em):
        eid = em.create("to_delete")
        em.delete(eid)
        assert em.get(eid) is None


class TestTaskLinking:
    def test_link_task(self, em):
        eid = em.create("exp")
        mgr = TrainingManager(workspace=Path(tempfile.mkdtemp()) / "ws")
        tid = mgr.create_task("train1", "yolov8n.pt", {"epochs": 10})
        em.link_task(eid, tid)
        tasks = em.get_tasks(eid)
        assert len(tasks) == 1
        assert tasks[0]["model_name"] == "train1"

    def test_unlink_task(self, em):
        eid = em.create("exp")
        mgr = TrainingManager(workspace=Path(tempfile.mkdtemp()) / "ws2")
        tid = mgr.create_task("t", "yolov8n.pt", {})
        em.link_task(eid, tid)
        em.unlink_task(eid, tid)
        assert em.get_tasks(eid) == []

    def test_link_duplicate_silent(self, em):
        eid = em.create("exp")
        mgr = TrainingManager(workspace=Path(tempfile.mkdtemp()) / "ws3")
        tid = mgr.create_task("t", "yolov8n.pt", {})
        em.link_task(eid, tid)
        em.link_task(eid, tid)  # 不应抛异常
        assert len(em.get_tasks(eid)) == 1

    def test_delete_clears_links(self, em):
        eid = em.create("exp")
        mgr = TrainingManager(workspace=Path(tempfile.mkdtemp()) / "ws4")
        tid = mgr.create_task("t", "yolov8n.pt", {})
        em.link_task(eid, tid)
        em.delete(eid)
        db = get_db()
        links = db.fetchall("SELECT * FROM experiment_task WHERE experiment_id = ?", (eid,))
        assert len(links) == 0


class TestComparison:
    def test_comparison_empty(self, em):
        comp = em.get_comparison([])
        assert comp["experiments"] == []

    def test_comparison_with_tasks(self, em):
        eid = em.create("exp_with_tasks")
        mgr = TrainingManager(workspace=Path(tempfile.mkdtemp()) / "ws5")
        tid = mgr.create_task("t1", "yolov8n.pt", {"epochs": 10})
        # 写 metrics 到日志
        task = mgr.get_task(tid)
        log_path = Path(task["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
            "   1/100      1.23G      1.456      2.345      1.234        123        640\n"
            "                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)\n"
            "                   all        100        500      0.789      0.723      0.782      0.567\n"
        )
        em.link_task(eid, tid)
        comp = em.get_comparison([eid])
        assert len(comp["experiments"]) == 1
        e = comp["experiments"][0]
        assert e["best_map50"] == 0.782
        assert e["task_count"] == 1
        assert len(e["tasks"]) == 1
        assert e["tasks"][0]["metrics"] != []

    def test_multi_experiment_comparison(self, em):
        e1 = em.create("exp_a")
        e2 = em.create("exp_b")
        comp = em.get_comparison([e1, e2])
        assert len(comp["experiments"]) == 2


class TestGetAvailableTasks:
    def test_returns_tasks(self, em):
        mgr = TrainingManager(workspace=Path(tempfile.mkdtemp()) / "ws6")
        mgr.create_task("a", "yolov8n.pt", {})
        mgr.create_task("b", "yolov8s.pt", {})
        tasks = em.get_available_tasks()
        assert len(tasks) == 2

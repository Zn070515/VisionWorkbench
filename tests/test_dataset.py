"""数据集模块单元测试."""

import tempfile
from pathlib import Path

import pytest

from vw.core.dataset.validator import validate_dataset, ValidationResult
from vw.infrastructure.database import reset_db


@pytest.fixture(autouse=True)
def _reset_db():
    reset_db()
    yield
    reset_db()


def create_yolo_dataset(tmp_path: Path, valid: bool = True):
    """创建临时 YOLO 数据集结构."""
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)

    import cv2
    import numpy as np

    for i in range(5):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(images / f"img_{i}.jpg"), img)

        if valid:
            labels_file = labels / f"img_{i}.txt"
            with open(labels_file, "w") as f:
                f.write(f"0 0.5 0.5 0.2 0.2\n")

    # 创建一个损坏的标签
    if not valid:
        with open(labels / "img_4.txt", "w") as f:
            f.write("invalid content\n")

    return tmp_path


def test_validation_result_empty():
    r = ValidationResult()
    assert r.valid is True
    assert r.errors == []
    d = r.to_dict()
    assert d["valid"] is True


def test_validate_empty_path():
    result = validate_dataset("/nonexistent/path")
    assert not result.valid
    assert any("路径不存在" in e for e in result.errors)


class TestValidateYOLODataset:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_valid_dataset(self):
        ds = create_yolo_dataset(Path(self.tmpdir) / "good_ds", valid=True)
        result = validate_dataset(str(ds))
        assert result.valid
        assert len(result.errors) == 0
        assert result.stats["total_images"] == 5
        assert result.stats["valid_labels"] == 5
        assert result.stats["num_classes"] == 1

    def test_dataset_with_broken_labels(self):
        ds = create_yolo_dataset(Path(self.tmpdir) / "bad_ds", valid=False)
        result = validate_dataset(str(ds))
        assert not result.valid
        assert result.stats["broken_labels"] >= 1

    def test_statistics_no_images(self):
        result = validate_dataset(str(Path(self.tmpdir) / "empty"))
        assert not result.valid


def test_registry_crud():
    import tempfile
    from vw.core.dataset.registry import DatasetRegistry
    from vw.infrastructure.database import get_db

    # 使用临时数据库
    db_path = Path(tempfile.mkdtemp()) / "test.db"
    db = get_db(db_path)

    reg = DatasetRegistry()

    ds_path = Path(tempfile.mkdtemp()) / "test_ds"
    ds_path.mkdir(parents=True)

    # 注册
    ds = reg.register("test", str(ds_path))
    assert ds["name"] == "test"
    assert ds["status"] == "registered"

    # 列表
    all_ds = reg.list_all()
    assert len(all_ds) == 1

    # 获取
    got = reg.get(ds["id"])
    assert got["name"] == "test"

    # 按路径获取
    got2 = reg.get_by_path(str(ds_path))
    assert got2 is not None

    # 更新状态
    reg.update_status(ds["id"], "validated", num_images=10)
    updated = reg.get(ds["id"])
    assert updated["status"] == "validated"
    assert updated["num_images"] == 10

    # 删除
    reg.delete(ds["id"])
    assert reg.get(ds["id"]) is None

    db.close()

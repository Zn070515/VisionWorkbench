"""数据集注册、列表、删除."""

from pathlib import Path
from typing import Optional

from vw.infrastructure.database import get_db


class DatasetRegistry:
    """数据集注册表."""

    def register(self, name: str, path: str | Path, ds_type: str = "yolo") -> dict:
        """注册数据集。返回创建的记录。"""
        db = get_db()
        abs_path = str(Path(path).resolve())
        row_id = db.insert("dataset", {
            "name": name,
            "path": abs_path,
            "type": ds_type,
        })
        return dict(db.fetchone("SELECT * FROM dataset WHERE id = ?", (row_id,)))

    def get(self, dataset_id: int) -> Optional[dict]:
        db = get_db()
        row = db.fetchone("SELECT * FROM dataset WHERE id = ?", (dataset_id,))
        return dict(row) if row else None

    def get_by_path(self, path: str | Path) -> Optional[dict]:
        abs_path = str(Path(path).resolve())
        db = get_db()
        row = db.fetchone("SELECT * FROM dataset WHERE path = ?", (abs_path,))
        return dict(row) if row else None

    def list_all(self) -> list[dict]:
        db = get_db()
        rows = db.fetchall(
            "SELECT * FROM dataset ORDER BY updated_at DESC"
        )
        return [dict(r) for r in rows]

    def update_status(self, dataset_id: int, status: str, **extra):
        db = get_db()
        data = {"status": status, "updated_at": _now(), **extra}
        db.update("dataset", data, "id = ?", (dataset_id,))

    def delete(self, dataset_id: int):
        db = get_db()
        db.execute("DELETE FROM dataset WHERE id = ?", (dataset_id,))
        db.commit()


def _now() -> str:
    from datetime import datetime as dt
    return dt.now().strftime("%Y-%m-%d %H:%M:%S")

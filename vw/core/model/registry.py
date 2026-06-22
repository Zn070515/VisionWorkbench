"""模型注册表 — 注册、版本管理、血统追踪."""

import json
import shutil
from pathlib import Path
from typing import Optional

from vw.infrastructure.database import get_db
from vw.infrastructure.filesystem import get_default_workspace


class ModelRegistry:
    """模型注册表."""

    def __init__(self, workspace: Optional[Path] = None):
        self.workspace = workspace or get_default_workspace()
        self._models_dir = self.workspace / "models"
        self._models_dir.mkdir(parents=True, exist_ok=True)

    # ── 模型 CRUD ──────────────────────────────────────────

    def register(
        self,
        name: str,
        file_path: str | Path,
        model_type: str = "yolo",
        task_type: Optional[str] = None,
        num_classes: Optional[int] = None,
        input_size: Optional[int] = None,
    ) -> int:
        """注册新模型，返回 model_id。"""
        db = get_db()
        row_id = db.insert("model", {
            "name": name,
            "type": model_type,
            "task_type": task_type,
            "num_classes": num_classes,
            "input_size": input_size,
        }, commit=False)
        # 首个版本默认 v1，成功后才一起提交
        self._add_version_internal(row_id, file_path, version=1)
        db.commit()
        return row_id

    def get(self, model_id: int) -> Optional[dict]:
        db = get_db()
        row = db.fetchone("SELECT * FROM model WHERE id = ?", (model_id,))
        return dict(row) if row else None

    def get_by_name(self, name: str) -> Optional[dict]:
        db = get_db()
        row = db.fetchone("SELECT * FROM model WHERE name = ?", (name,))
        return dict(row) if row else None

    def list_all(self) -> list[dict]:
        db = get_db()
        rows = db.fetchall("SELECT * FROM model ORDER BY created_at DESC")
        return [dict(r) for r in rows]

    def delete(self, model_id: int):
        db = get_db()
        # 级联删除版本记录，保留文件
        db.execute("DELETE FROM model_version WHERE model_id = ?", (model_id,))
        db.execute("DELETE FROM model WHERE id = ?", (model_id,))
        db.commit()

    # ── 版本管理 ───────────────────────────────────────────

    def add_version(
        self,
        model_id: int,
        file_path: str | Path,
        training_task_id: Optional[int] = None,
        dataset_id: Optional[int] = None,
        metrics: Optional[dict] = None,
        notes: Optional[str] = None,
    ) -> int:
        """为已有模型添加新版本，返回 version_id。"""
        existing = self.get_versions(model_id)
        next_ver = max((v["version"] for v in existing), default=0) + 1
        return self._add_version_internal(
            model_id, file_path, version=next_ver,
            training_task_id=training_task_id, dataset_id=dataset_id,
            metrics=metrics, notes=notes,
        )

    def _add_version_internal(
        self,
        model_id: int,
        file_path: str | Path,
        version: int = 1,
        training_task_id: Optional[int] = None,
        dataset_id: Optional[int] = None,
        metrics: Optional[dict] = None,
        notes: Optional[str] = None,
    ) -> int:
        db = get_db()
        src = Path(file_path)
        # 拷贝到模型目录
        model_dir = self._models_dir / str(model_id)
        model_dir.mkdir(parents=True, exist_ok=True)
        dest = model_dir / f"v{version}_{src.name}"
        if not dest.exists() or dest.resolve() != src.resolve():
            shutil.copy2(src, dest)

        metrics = metrics or {}
        row_id = db.insert("model_version", {
            "model_id": model_id,
            "version": version,
            "file_path": str(dest),
            "file_size": dest.stat().st_size if dest.exists() else None,
            "training_task_id": training_task_id,
            "dataset_id": dataset_id,
            "map50": metrics.get("map50"),
            "map50_95": metrics.get("map50_95"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "notes": notes,
        })
        return row_id

    def get_versions(self, model_id: int) -> list[dict]:
        db = get_db()
        rows = db.fetchall(
            "SELECT * FROM model_version WHERE model_id = ? ORDER BY version DESC",
            (model_id,),
        )
        return [dict(r) for r in rows]

    def get_latest_version(self, model_id: int) -> Optional[dict]:
        versions = self.get_versions(model_id)
        return versions[0] if versions else None

    # ── 血统追踪 ───────────────────────────────────────────

    def get_lineage(self, model_id: int) -> dict:
        """返回模型的血统链路：模型 ← 训练 ← 数据集。"""
        model = self.get(model_id)
        if not model:
            return {}
        versions = self.get_versions(model_id)

        lineage = {
            "model": model,
            "versions": [],
        }

        for v in versions:
            v_info = dict(v)
            # 训练任务
            if v["training_task_id"]:
                db = get_db()
                train = db.fetchone(
                    "SELECT * FROM training_task WHERE id = ?",
                    (v["training_task_id"],),
                )
                if train:
                    train_dict = dict(train)
                    if isinstance(train_dict.get("config"), str):
                        try:
                            train_dict["config"] = json.loads(train_dict["config"])
                        except Exception:
                            pass
                    v_info["training_task"] = train_dict
                    # 数据集
                    if train_dict.get("dataset_id"):
                        ds = db.fetchone(
                            "SELECT * FROM dataset WHERE id = ?",
                            (train_dict["dataset_id"],),
                        )
                        if ds:
                            v_info["dataset"] = dict(ds)
            lineage["versions"].append(v_info)

        return lineage

    # ── 默认模型 ───────────────────────────────────────────

    def set_default(self, model_id: int):
        db = get_db()
        db.execute("UPDATE model SET is_default = 0", ())
        db.update("model", {"is_default": 1}, "id = ?", (model_id,))

    def get_default(self) -> Optional[dict]:
        db = get_db()
        row = db.fetchone("SELECT * FROM model WHERE is_default = 1")
        return dict(row) if row else None

    # ── 训练完成自动注册 ────────────────────────────────────

    def register_from_training(
        self,
        training_task_id: int,
        model_file: str | Path,
        metrics: Optional[dict] = None,
    ) -> int:
        """训练完成后自动注册模型。若同名模型已存在则追加版本。"""
        db = get_db()
        train = db.fetchone(
            "SELECT * FROM training_task WHERE id = ?", (training_task_id,)
        )
        if not train:
            raise ValueError(f"训练任务不存在: {training_task_id}")

        train = dict(train)
        name = train["model_name"]

        existing = self.get_by_name(name)
        if existing:
            model_id = existing["id"]
            self.add_version(
                model_id, model_file,
                training_task_id=training_task_id,
                dataset_id=train.get("dataset_id"),
                metrics=metrics,
            )
        else:
            # register() 已创建 v1，用 training info 更新该版本
            model_id = self.register(
                name=name,
                file_path=model_file,
                model_type="yolo",
            )
            db = get_db()
            db.update("model_version", {
                "training_task_id": training_task_id,
                "dataset_id": train.get("dataset_id"),
                "map50": (metrics or {}).get("map50"),
                "map50_95": (metrics or {}).get("map50_95"),
                "precision": (metrics or {}).get("precision"),
                "recall": (metrics or {}).get("recall"),
            }, "model_id = ? AND version = 1", (model_id,))
        return model_id

    # ── 导出 ───────────────────────────────────────────────

    def export(self, model_id: int, dest_dir: str | Path):
        """导出模型的所有版本到目标目录。"""
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        for v in self.get_versions(model_id):
            src = Path(v["file_path"])
            if src.exists():
                shutil.copy2(src, dest / src.name)

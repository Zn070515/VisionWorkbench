"""实验管理 — 创建、关联训练任务、指标对比."""

import json
from typing import Optional

from vw.infrastructure.database import get_db
from vw.core.training.manager import TrainingManager


class ExperimentManager:
    """实验管理器."""

    def create(
        self,
        name: str,
        description: Optional[str] = None,
        hypothesis: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> int:
        """创建实验，返回 experiment_id。"""
        db = get_db()
        row_id = db.insert("experiment", {
            "name": name,
            "description": description,
            "hypothesis": hypothesis,
            "tags": json.dumps(tags or []),
        })
        return row_id

    def get(self, experiment_id: int) -> Optional[dict]:
        db = get_db()
        row = db.fetchone("SELECT * FROM experiment WHERE id = ?", (experiment_id,))
        if not row:
            return None
        d = dict(row)
        d["tags"] = _parse_tags(d.get("tags"))
        return d

    def list_all(self) -> list[dict]:
        db = get_db()
        rows = db.fetchall("SELECT * FROM experiment ORDER BY updated_at DESC")
        result = []
        for r in rows:
            d = dict(r)
            d["tags"] = _parse_tags(d.get("tags"))
            result.append(d)
        return result

    def update(self, experiment_id: int, **fields):
        """更新实验字段（description, hypothesis, conclusion, tags, name）。"""
        db = get_db()
        if "tags" in fields and isinstance(fields["tags"], list):
            fields["tags"] = json.dumps(fields["tags"])
        fields["updated_at"] = _now()
        db.update("experiment", fields, "id = ?", (experiment_id,))

    def delete(self, experiment_id: int):
        db = get_db()
        db.execute("DELETE FROM experiment_task WHERE experiment_id = ?", (experiment_id,))
        db.execute("DELETE FROM experiment WHERE id = ?", (experiment_id,))
        db.commit()

    # ── 任务关联 ───────────────────────────────────────────

    def link_task(self, experiment_id: int, task_id: int):
        db = get_db()
        try:
            db.insert("experiment_task", {
                "experiment_id": experiment_id,
                "task_id": task_id,
            })
        except Exception:
            pass  # 已存在则忽略

    def unlink_task(self, experiment_id: int, task_id: int):
        db = get_db()
        db.execute(
            "DELETE FROM experiment_task WHERE experiment_id = ? AND task_id = ?",
            (experiment_id, task_id),
        )
        db.commit()

    def get_tasks(self, experiment_id: int) -> list[dict]:
        db = get_db()
        rows = db.fetchall(
            """SELECT t.* FROM training_task t
               JOIN experiment_task et ON t.id = et.task_id
               WHERE et.experiment_id = ?
               ORDER BY t.created_at DESC""",
            (experiment_id,),
        )
        return [dict(r) for r in rows]

    # ── 指标对比 ───────────────────────────────────────────

    def get_comparison(self, experiment_ids: list[int]) -> dict:
        """返回多实验指标对比数据。"""
        tm = TrainingManager()
        experiments = []

        for eid in experiment_ids:
            exp = self.get(eid)
            if not exp:
                continue
            tasks = self.get_tasks(eid)
            task_summaries = []

            best_map50 = 0.0
            best_map50_95 = 0.0

            for t in tasks:
                metrics = tm.get_metrics(t["id"])
                if metrics:
                    max_m = max(metrics, key=lambda m: m.get("map50", 0))
                    m50 = max_m.get("map50", 0) or 0
                    m95 = max_m.get("map50_95", 0) or 0
                    if m50 > best_map50:
                        best_map50 = m50
                    if m95 > best_map50_95:
                        best_map50_95 = m95
                    task_summaries.append({
                        "task_id": t["id"],
                        "model_name": t["model_name"],
                        "base_model": t["base_model"],
                        "status": t["status"],
                        "best_map50": m50,
                        "best_map50_95": m95,
                        "best_precision": max_m.get("precision", 0) or 0,
                        "best_recall": max_m.get("recall", 0) or 0,
                        "epochs_completed": len(metrics),
                        "metrics": metrics,
                    })
                else:
                    task_summaries.append({
                        "task_id": t["id"],
                        "model_name": t["model_name"],
                        "base_model": t["base_model"],
                        "status": t["status"],
                        "best_map50": 0,
                        "best_map50_95": 0,
                        "best_precision": 0,
                        "best_recall": 0,
                        "epochs_completed": 0,
                        "metrics": [],
                    })

            experiments.append({
                "id": exp["id"],
                "name": exp["name"],
                "description": exp.get("description"),
                "hypothesis": exp.get("hypothesis"),
                "conclusion": exp.get("conclusion"),
                "tags": exp.get("tags", []),
                "task_count": len(tasks),
                "best_map50": best_map50,
                "best_map50_95": best_map50_95,
                "tasks": task_summaries,
            })

        return {"experiments": experiments}

    def get_available_tasks(self) -> list[dict]:
        """返回可关联的训练任务列表（已完成的优先）。"""
        tm = TrainingManager()
        return tm.list_tasks()


def _parse_tags(raw) -> list[str]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return [raw] if raw else []
    return []


def _now() -> str:
    from datetime import datetime as dt
    return dt.now().strftime("%Y-%m-%d %H:%M:%S")

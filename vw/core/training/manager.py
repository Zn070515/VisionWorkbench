"""训练任务管理 — 启动、监控、停止 YOLO 训练."""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from vw.infrastructure.database import get_db
from vw.infrastructure.filesystem import get_default_workspace


def _find_yolo() -> str:
    """查找 yolo CLI 可执行文件路径。"""
    yolo = shutil.which("yolo")
    if yolo:
        return yolo
    name = "yolo.exe" if sys.platform == "win32" else "yolo"
    scripts = Path(sys.executable).parent / name
    if scripts.exists():
        return str(scripts)
    raise FileNotFoundError(
        "找不到 yolo CLI。请确保 ultralytics 已安装: pip install ultralytics"
    )


_METRICS_RE = re.compile(
    r"^\s*(\d+)/\d+\s+[\d.]+\w+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+\s+\d+"
)
_MAP_RE = re.compile(
    r"^\s*all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
)


class TrainingManager:
    """YOLO 训练任务管理器."""

    def __init__(self, workspace: Optional[Path] = None):
        self.workspace = workspace or get_default_workspace()
        self._training_dir = self.workspace / "training"
        self._training_dir.mkdir(parents=True, exist_ok=True)

    def create_task(
        self,
        model_name: str,
        base_model: str,
        config: dict,
        dataset_id: Optional[int] = None,
    ) -> int:
        """创建训练任务记录，返回 task_id。"""
        db = get_db()
        task_dir = self._training_dir / str(int(time.time()))
        log_path = str(task_dir / "train.log")
        results_path = str(task_dir / "results")
        row_id = db.insert("training_task", {
            "dataset_id": dataset_id,
            "model_name": model_name,
            "base_model": base_model,
            "config": json.dumps(config),
            "log_path": log_path,
            "results_path": results_path,
            "status": "pending",
        })
        # DB 插入成功后再创建磁盘目录，避免孤儿目录
        task_dir.mkdir(parents=True, exist_ok=True)
        return row_id

    def launch(self, task_id: int) -> bool:
        """启动训练子进程。成功返回 True。"""
        db = get_db()
        task = db.fetchone("SELECT * FROM training_task WHERE id = ?", (task_id,))
        if not task:
            return False

        task = dict(task)
        if task["status"] in ("running",):
            return False

        config = json.loads(task["config"])
        log_path = task["log_path"]

        cmd = self._build_command(task, config)

        log_dir = Path(log_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(log_dir),
                start_new_session=True,
            )
        finally:
            log_file.close()

        db.update("training_task", {
            "status": "running",
            "pid": process.pid,
            "started_at": _now(),
        }, "id = ?", (task_id,))

        return True

    def stop(self, task_id: int) -> bool:
        """停止训练任务。"""
        db = get_db()
        task = db.fetchone("SELECT * FROM training_task WHERE id = ?", (task_id,))
        if not task:
            return False
        task = dict(task)
        if task["status"] != "running":
            return False
        pid = task["pid"]
        if pid:
            try:
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                                   capture_output=True)
                else:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
            except Exception:
                pass
        db.update("training_task", {
            "status": "stopped",
            "finished_at": _now(),
        }, "id = ?", (task_id,))
        return True

    def get_task(self, task_id: int) -> Optional[dict]:
        db = get_db()
        row = db.fetchone("SELECT * FROM training_task WHERE id = ?", (task_id,))
        return dict(row) if row else None

    def get_log(self, task_id: int, tail: int = 80) -> str:
        """读取训练日志的最后 N 行。"""
        task = self.get_task(task_id)
        if not task or not task["log_path"]:
            return ""
        log_path = Path(task["log_path"])
        if not log_path.exists():
            return ""
        try:
            with open(log_path, "r", errors="replace") as f:
                lines = f.readlines()
                return "".join(lines[-tail:])
        except Exception:
            return ""

    def get_metrics(self, task_id: int) -> list[dict]:
        """从日志解析训练指标，返回 epoch 级别的 metric 列表。"""
        task = self.get_task(task_id)
        if not task or not task["log_path"]:
            return []
        log_path = Path(task["log_path"])
        if not log_path.exists():
            return []

        metrics = []
        current_epoch = None
        try:
            with open(log_path, "r", errors="replace") as f:
                for line in f:
                    m = _METRICS_RE.match(line)
                    if m:
                        current_epoch = {
                            "epoch": int(m.group(1)),
                            "box_loss": float(m.group(2)),
                            "cls_loss": float(m.group(3)),
                            "dfl_loss": float(m.group(4)),
                        }
                        continue
                    m_map = _MAP_RE.match(line)
                    if m_map and current_epoch and current_epoch["epoch"] not in {e["epoch"] for e in metrics}:
                        current_epoch["precision"] = float(m_map.group(1))
                        current_epoch["recall"] = float(m_map.group(2))
                        current_epoch["map50"] = float(m_map.group(3))
                        current_epoch["map50_95"] = float(m_map.group(4))
                        metrics.append(current_epoch)
                        current_epoch = None
        except Exception:
            pass
        return metrics

    def list_tasks(self, status: Optional[str] = None) -> list[dict]:
        db = get_db()
        if status:
            rows = db.fetchall(
                "SELECT * FROM training_task WHERE status = ? ORDER BY created_at DESC",
                (status,),
            )
        else:
            rows = db.fetchall("SELECT * FROM training_task ORDER BY created_at DESC")
        return [dict(r) for r in rows]

    def delete_task(self, task_id: int):
        task = self.get_task(task_id)
        if task and task["status"] == "running":
            self.stop(task_id)
        db = get_db()
        db.execute("DELETE FROM training_task WHERE id = ?", (task_id,))
        db.commit()

    def poll_completion(self, task_id: int) -> Optional[str]:
        """检查子进程是否已结束。返回新状态或 None。"""
        task = self.get_task(task_id)
        if not task or task["status"] != "running":
            return None
        pid = task["pid"]
        if pid is None:
            return None
        exited = False
        exit_code = None
        try:
            if sys.platform == "win32":
                import ctypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.OpenProcess(0x0400, False, pid)
                if handle:
                    code = ctypes.c_ulong()
                    kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
                    kernel32.CloseHandle(handle)
                    if code.value != 259:  # STILL_ACTIVE
                        exited = True
                        exit_code = code.value
            else:
                wpid, status = os.waitpid(pid, os.WNOHANG)
                if wpid == pid:
                    exited = True
                    exit_code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else -1
        except Exception:
            pass

        if exited:
            status = "completed" if exit_code == 0 else "failed"
            db = get_db()
            db.update("training_task", {
                "status": status,
                "finished_at": _now(),
            }, "id = ?", (task_id,))
            return status
        return None

    def _build_command(self, task: dict, config: dict) -> list:
        data_path = config.get("data", "")
        model_path = config.get("model", task["base_model"])
        epochs = config.get("epochs", 100)
        imgsz = config.get("imgsz", 640)
        batch = config.get("batch", 16)
        device = config.get("device", "cpu")
        name = task.get("model_name", "train")
        project = str(Path(task["results_path"]).parent)

        return [
            _find_yolo(), "train",
            f"model={model_path}",
            f"data={data_path}",
            f"epochs={epochs}",
            f"imgsz={imgsz}",
            f"batch={batch}",
            f"device={device}",
            f"project={project}",
            f"name={name}",
        ]


def _now() -> str:
    from datetime import datetime as dt
    return dt.now().strftime("%Y-%m-%d %H:%M:%S")

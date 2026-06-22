"""SQLite 数据库连接管理和迁移."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS dataset (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    path        TEXT NOT NULL UNIQUE,
    type        TEXT NOT NULL DEFAULT 'yolo',
    num_classes INTEGER DEFAULT 0,
    num_images  INTEGER DEFAULT 0,
    status      TEXT NOT NULL DEFAULT 'registered',
    metadata    TEXT DEFAULT '{}',
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS training_task (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id  INTEGER REFERENCES dataset(id),
    model_name  TEXT NOT NULL,
    base_model  TEXT NOT NULL,
    config      TEXT NOT NULL DEFAULT '{}',
    status      TEXT NOT NULL DEFAULT 'pending',
    pid         INTEGER,
    log_path    TEXT,
    results_path TEXT,
    started_at  TEXT,
    finished_at TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS model (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    type        TEXT NOT NULL DEFAULT 'yolo',
    task_type   TEXT,
    num_classes INTEGER,
    input_size  INTEGER,
    is_default  INTEGER DEFAULT 0,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS model_version (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id        INTEGER NOT NULL REFERENCES model(id),
    version         INTEGER NOT NULL,
    file_path       TEXT NOT NULL,
    file_size       INTEGER,
    training_task_id INTEGER REFERENCES training_task(id),
    dataset_id      INTEGER REFERENCES dataset(id),
    map50           REAL,
    map50_95        REAL,
    precision       REAL,
    recall          REAL,
    notes           TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS experiment (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    description TEXT,
    hypothesis  TEXT,
    conclusion  TEXT,
    tags        TEXT DEFAULT '[]',
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS experiment_task (
    experiment_id INTEGER NOT NULL REFERENCES experiment(id),
    task_id       INTEGER NOT NULL REFERENCES training_task(id),
    PRIMARY KEY (experiment_id, task_id)
);

CREATE TABLE IF NOT EXISTS evaluation (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version_id INTEGER NOT NULL REFERENCES model_version(id),
    dataset_id      INTEGER REFERENCES dataset(id),
    type            TEXT NOT NULL DEFAULT 'image',
    input_path      TEXT NOT NULL,
    results         TEXT DEFAULT '{}',
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_dataset_status ON dataset(status);
CREATE INDEX IF NOT EXISTS idx_training_task_status ON training_task(status);
CREATE INDEX IF NOT EXISTS idx_model_version_model ON model_version(model_id);
CREATE INDEX IF NOT EXISTS idx_experiment_task_exp ON experiment_task(experiment_id);
"""


class Database:
    """SQLite 数据库管理器."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def init(self):
        """创建所有表。"""
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def execute(self, sql: str, params=()):
        return self.conn.execute(sql, params)

    def executemany(self, sql: str, params_list):
        return self.conn.executemany(sql, params_list)

    def commit(self):
        self.conn.commit()

    def fetchone(self, sql: str, params=()):
        return self.conn.execute(sql, params).fetchone()

    def fetchall(self, sql: str, params=()):
        return self.conn.execute(sql, params).fetchall()

    def insert(self, table: str, data: dict, commit: bool = True) -> int:
        columns = ", ".join(data.keys())
        placeholders = ", ".join("?" for _ in data)
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        cursor = self.conn.execute(sql, tuple(data.values()))
        if commit:
            self.conn.commit()
        return cursor.lastrowid

    def update(self, table: str, data: dict, where: str, params=(),
               commit: bool = True) -> int:
        sets = ", ".join(f"{k} = ?" for k in data)
        sql = f"UPDATE {table} SET {sets} WHERE {where}"
        values = tuple(data.values()) + tuple(params)
        cursor = self.conn.execute(sql, values)
        if commit:
            self.conn.commit()
        return cursor.rowcount


_db_instance: Optional[Database] = None


def get_db(path: str | Path = None) -> Database:
    global _db_instance
    if _db_instance is None:
        if path is None:
            path = Path.home() / "visionworkbench" / "vw.db"
        _db_instance = Database(path)
        _db_instance.init()
    return _db_instance


def reset_db():
    """关闭并重置数据库单例（测试用）."""
    global _db_instance
    if _db_instance:
        _db_instance.close()
    _db_instance = None

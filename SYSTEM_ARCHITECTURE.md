# SYSTEM_ARCHITECTURE.md — VisionWorkbench

## 概述

VisionWorkbench 采用**单体应用 + 模块化内核**架构。所有功能在一个 Python 进程中运行，通过明确的模块边界保持代码组织性，同时避免分布式系统的复杂性。

---

## 1. 架构原则

| 原则 | 说明 |
|------|------|
| **单进程优先** | 不引入微服务。训练可 fork 子进程但功能在同一进程 |
| **文件系统即存储** | 模型、数据集以文件形式存储。SQLite 只存元数据 |
| **模块松耦合** | 每个模块有明确的输入/输出接口 |
| **UI 与逻辑分离** | 核心逻辑与 Streamlit UI 层分目录组织 |
| **渐进复杂度** | v0.1 简单架构，v1.0 才引入完整模块化 |
| **AI 可读性** | 目录结构、配置文件、数据库模式设计为 AI 可理解 |

---

## 2. 技术选型

### 2.1 选型理由

```
需求                    →  选型           →  理由
─────────────────────────────────────────────────────
Web UI                 →  Streamlit      →  Python 原生，最快出 UI
ML 训练                 →  Ultralytics    →  YOLO 标准框架
图像处理                →  OpenCV         →  行业标准
数据库                  →  SQLite         →  零配置，单文件
绘图                    →  Plotly         →  交互式图表，Streamlit 原生支持
配置                    →  YAML           →  人类可读，AI 可写
包管理                  →  uv             →  快速，可靠
```

### 2.2 技术栈一览

| 层 | 技术 | 版本要求 |
|----|------|---------|
| 语言 | Python | 3.10+ |
| UI | Streamlit | 1.28+ |
| ML 框架 | Ultralytics | 8.2+ |
| 深度学习 | PyTorch | 2.0+ |
| 视觉库 | OpenCV | 4.8+ |
| 绘图 | Plotly | 5.18+ |
| 数据库 | SQLite | 3.35+（内置于 Python） |
| 数据处理 | NumPy, Pandas | - |
| 图像处理 | Pillow | 10.0+ |
| 配置 | PyYAML | 6.0+ |

### 2.3 明确不使用的技术

| 技术 | 不使用的理由 |
|------|-------------|
| Docker | 增加复杂度，学生不需要 |
| FastAPI/Flask | Streamlit 自带 Web 服务 |
| PostgreSQL/MySQL | 需要额外安装，SQLite 足够 |
| Redis/RabbitMQ | 单进程不需要消息队列 |
| React/Vue | 增加前后端分离复杂度 |

---

## 3. 系统分层

```
┌─────────────────────────────────────────────┐
│              Presentation Layer              │
│         Streamlit Pages (UI)                │
├─────────────────────────────────────────────┤
│              Application Layer               │
│         Module Controllers                   │
├─────────────────────────────────────────────┤
│                Core Layer                    │
│    Workspace / Model / Dataset / Training    │
├─────────────────────────────────────────────┤
│            Infrastructure Layer              │
│      Database / FileSystem / Config          │
└─────────────────────────────────────────────┘
```

### 3.1 各层职责

**Presentation Layer（展示层）：**
- Streamlit 页面文件
- UI 布局和交互逻辑
- 调用 Application Layer 的接口

**Application Layer（应用层）：**
- 各模块的业务逻辑编排
- 模块间协调
- 用户操作的验证和处理

**Core Layer（核心层）：**
- 核心领域模型和业务规则
- 训练任务管理
- 模型加载和推理
- 数据集校验逻辑

**Infrastructure Layer（基础设施层）：**
- SQLite 数据库访问
- 文件系统操作
- 配置管理
- 日志记录

---

## 4. 目录结构

```
visionworkbench/
├── vw/
│   ├── __init__.py
│   ├── main.py                  # 入口：streamlit run
│   ├── cli.py                   # CLI 入口：vw start
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py          # 全局配置
│   │   └── defaults.yaml        # 默认配置值
│   │
│   ├── infrastructure/
│   │   ├── __init__.py
│   │   ├── database.py          # SQLite 连接和迁移
│   │   ├── filesystem.py        # 工作区目录管理
│   │   └── logging.py           # 日志配置
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── workspace.py         # 工作区管理
│   │   ├── dataset/
│   │   │   ├── __init__.py
│   │   │   ├── registry.py      # 数据集注册
│   │   │   ├── validator.py     # 数据集校验
│   │   │   └── statistics.py    # 数据集统计
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   ├── launcher.py      # 训练启动器
│   │   │   ├── monitor.py       # 训练监控
│   │   │   └── parser.py        # 训练结果解析
│   │   ├── model/
│   │   │   ├── __init__.py
│   │   │   ├── registry.py      # 模型注册表
│   │   │   ├── lineage.py       # 模型血统追踪
│   │   │   └── inference.py     # 推理引擎
│   │   ├── experiment/
│   │   │   ├── __init__.py
│   │   │   ├── manager.py       # 实验管理
│   │   │   └── comparator.py    # 实验对比
│   │   └── evaluation/
│   │       ├── __init__.py
│   │       ├── runner.py        # 评估运行器
│   │       └── analyzer.py      # 错误分析
│   │
│   ├── app/
│   │   ├── __init__.py
│   │   ├── home.py              # 概览页
│   │   ├── dataset_page.py      # 数据集页面
│   │   ├── training_page.py     # 训练页面
│   │   ├── experiment_page.py   # 实验页面
│   │   ├── model_page.py        # 模型页面
│   │   ├── evaluation_page.py   # 评估页面
│   │   └── demo_page.py         # 演示页面
│   │
│   ├── agent/                   # AI Agent 接口层（v1.0+）
│   │   ├── __init__.py
│   │   ├── base.py              # Agent 基类
│   │   └── tools.py             # Agent 可用的工具函数
│   │
│   └── utils/
│       ├── __init__.py
│       ├── image.py             # 图像工具函数
│       ├── yolo.py              # YOLO 工具函数
│       └── metrics.py           # 指标计算工具
│
├── tests/
│   ├── test_dataset/
│   ├── test_training/
│   ├── test_model/
│   ├── test_experiment/
│   └── test_evaluation/
│
├── docs/
│   └── user_guide/              # 用户文档
│
├── pyproject.toml
├── uv.lock
├── GOAL.md
├── PRODUCT_ANALYSIS.md
├── COMPETITOR_ANALYSIS.md
├── MVP_SPEC.md
└── SYSTEM_ARCHITECTURE.md
```

---

## 5. 数据库设计

### 5.1 核心表结构

```sql
-- 工作区
CREATE TABLE workspace (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    path        TEXT NOT NULL UNIQUE,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 数据集
CREATE TABLE dataset (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    path        TEXT NOT NULL,
    type        TEXT NOT NULL DEFAULT 'yolo',  -- yolo, coco, voc
    num_classes INTEGER DEFAULT 0,
    num_images  INTEGER DEFAULT 0,
    status      TEXT NOT NULL DEFAULT 'registered',  -- registered, validated, error
    metadata    TEXT DEFAULT '{}',  -- JSON
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 数据集版本
CREATE TABLE dataset_version (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id  INTEGER NOT NULL REFERENCES dataset(id),
    version     INTEGER NOT NULL,
    description TEXT,
    snapshot_path TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 训练任务
CREATE TABLE training_task (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id      INTEGER REFERENCES dataset(id),
    model_name      TEXT NOT NULL,
    base_model      TEXT NOT NULL,  -- e.g. yolov8n.pt
    config          TEXT NOT NULL DEFAULT '{}',  -- JSON of hyperparams
    status          TEXT NOT NULL DEFAULT 'pending',
        -- pending, running, completed, failed, stopped
    pid             INTEGER,
    log_path        TEXT,
    results_path    TEXT,
    started_at      TEXT,
    finished_at     TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 训练指标（从训练结果解析）
CREATE TABLE training_metric (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id         INTEGER NOT NULL REFERENCES training_task(id),
    epoch           INTEGER NOT NULL,
    train_loss      REAL,
    val_loss        REAL,
    map50           REAL,
    map50_95        REAL,
    precision       REAL,
    recall          REAL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 模型
CREATE TABLE model (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    type            TEXT NOT NULL DEFAULT 'yolo',  -- yolo, other
    task_type       TEXT,  -- detect, segment, classify, pose
    num_classes     INTEGER,
    input_size      INTEGER,
    is_default      INTEGER DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 模型版本
CREATE TABLE model_version (
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

-- 实验
CREATE TABLE experiment (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    description     TEXT,
    hypothesis      TEXT,
    conclusion      TEXT,
    tags            TEXT DEFAULT '[]',  -- JSON array
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 实验关联训练任务
CREATE TABLE experiment_task (
    experiment_id   INTEGER NOT NULL REFERENCES experiment(id),
    task_id         INTEGER NOT NULL REFERENCES training_task(id),
    PRIMARY KEY (experiment_id, task_id)
);

-- 评估
CREATE TABLE evaluation (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version_id INTEGER NOT NULL REFERENCES model_version(id),
    dataset_id      INTEGER REFERENCES dataset(id),
    type            TEXT NOT NULL DEFAULT 'image',  -- image, video, batch
    input_path      TEXT NOT NULL,
    results         TEXT DEFAULT '{}',  -- JSON
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 评估错误样本
CREATE TABLE error_sample (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_id   INTEGER NOT NULL REFERENCES evaluation(id),
    image_path      TEXT NOT NULL,
    ground_truth    TEXT,  -- JSON
    prediction      TEXT,  -- JSON
    error_type      TEXT,  -- false_positive, false_negative, misclassification
    confidence      REAL,
    note            TEXT
);
```

### 5.2 索引策略

```sql
CREATE INDEX idx_training_task_status ON training_task(status);
CREATE INDEX idx_model_version_model ON model_version(model_id);
CREATE INDEX idx_model_version_training ON model_version(training_task_id);
CREATE INDEX idx_experiment_task_exp ON experiment_task(experiment_id);
CREATE INDEX idx_evaluation_model ON evaluation(model_version_id);
CREATE INDEX idx_error_sample_eval ON error_sample(evaluation_id);
```

---

## 6. 模块接口设计

### 6.1 模块间通信

模块间通过直接的 Python 函数调用通信，不经过网络或消息队列：

```python
# 示例：训练完成后自动注册模型
# training/launcher.py → model/registry.py

class TrainingLauncher:
    def __init__(self, model_registry: ModelRegistry):
        self.model_registry = model_registry

    def on_training_complete(self, task: TrainingTask):
        # 解析训练结果
        metrics = parse_training_results(task.results_path)
        # 注册模型
        self.model_registry.register(
            name=task.model_name,
            file_path=f"{task.results_path}/weights/best.pt",
            training_task_id=task.id,
            dataset_id=task.dataset_id,
            metrics=metrics
        )
```

### 6.2 核心接口

```python
# Dataset Registry
class DatasetRegistry:
    def register(name: str, path: str, type: str = "yolo") -> Dataset
    def get(id: int) -> Dataset
    def list_all() -> list[Dataset]
    def validate(id: int) -> ValidationReport
    def get_statistics(id: int) -> DatasetStatistics
    def delete(id: int) -> None

# Training Launcher
class TrainingLauncher:
    def configure(params: TrainingConfig) -> TrainingTask
    def start(task_id: int) -> None
    def stop(task_id: int) -> None
    def get_status(task_id: int) -> TaskStatus
    def get_logs(task_id: int) -> Iterator[str]
    def list_tasks(status: str = None) -> list[TrainingTask]
    def get_metrics(task_id: int) -> list[TrainingMetric]

# Model Registry
class ModelRegistry:
    def register(name: str, file_path: str, **metadata) -> ModelVersion
    def get(id: int) -> Model
    def get_version(version_id: int) -> ModelVersion
    def list_models() -> list[Model]
    def get_lineage(model_id: int) -> LineageGraph
    def compare(version_ids: list[int]) -> ComparisonResult
    def set_default(model_id: int) -> None
    def get_default() -> ModelVersion

# Experiment Manager
class ExperimentManager:
    def create(name: str, task_ids: list[int]) -> Experiment
    def get(id: int) -> Experiment
    def list_all() -> list[Experiment]
    def add_note(id: int, note: str) -> None
    def set_conclusion(id: int, conclusion: str) -> None
    def compare(experiment_ids: list[int]) -> ComparisonResult

# Evaluation Runner
class EvaluationRunner:
    def run_image(model_id: int, image_path: str) -> EvaluationResult
    def run_video(model_id: int, video_path: str) -> EvaluationResult
    def run_batch(model_id: int, dataset_id: int) -> EvaluationResult
    def analyze_errors(eval_id: int) -> ErrorAnalysis

# Demo Runner
class DemoRunner:
    def demo_image(model_id: int, image_path: str) -> None
    def demo_video(model_id: int, video_path: str) -> None
    def demo_webcam(model_id: int) -> None
```

---

## 7. 数据流

### 7.1 完整工作流数据流

```
用户注册数据集
    │
    ▼
DatasetRegistry.register() ──→ SQLite (dataset 表)
    │                              └── 文件系统 (datasets/ 目录)
    ▼
DatasetRegistry.validate() ──→ 校验报告
    │
    ▼
用户配置训练
    │
    ▼
TrainingLauncher.configure() ──→ SQLite (training_task 表)
    │
    ▼
TrainingLauncher.start() ──→ 子进程 (ultralytics train)
    │                              └── 文件系统 (training/{id}/logs)
    ▼
TrainingMonitor 监听 ──→ SQLite (training_metric 表)
    │
    ▼
训练完成 ──→ ModelRegistry.register() ──→ SQLite (model, model_version 表)
    │                                           └── 文件系统 (models/ 目录)
    ▼
用户创建实验
    │
    ▼
ExperimentManager.create() ──→ SQLite (experiment, experiment_task 表)
    │
    ▼
EvaluationRunner.run_*() ──→ SQLite (evaluation, error_sample 表)
    │                              └── 文件系统 (evaluations/ 目录)
    ▼
DemoRunner.demo_*() ──→ Streamlit 实时显示
```

### 7.2 模型血统追踪

```
Dataset (v2)
  │
  └── TrainingTask (#42, epochs=100, batch=16)
        │
        └── Model "my_detector" v3
              │
              ├── Experiment "aug_test" ──→ 对比 "lr_tuning"
              │
              ├── Evaluation (#102) ──→ mAP50: 0.892, 错误: 12/200
              │
              └── Demo ──→ 实际部署使用
```

---

## 8. UI 架构

### 8.1 Streamlit 页面结构

```python
# main.py
import streamlit as st

def main():
    st.set_page_config(
        page_title="VisionWorkbench",
        page_icon="🔬",
        layout="wide"
    )

    # 侧边栏导航
    page = st.sidebar.radio(
        "导航",
        ["概览", "数据集", "训练", "实验", "模型", "评估", "演示"]
    )

    # 路由到对应页面
    if page == "概览":
        home_page.render()
    elif page == "数据集":
        dataset_page.render()
    elif page == "训练":
        training_page.render()
    elif page == "实验":
        experiment_page.render()
    elif page == "模型":
        model_page.render()
    elif page == "评估":
        evaluation_page.render()
    elif page == "演示":
        demo_page.render()
```

### 8.2 Session State 管理

```python
# 全局状态
class AppState:
    workspace: Workspace
    default_model: ModelVersion | None

# Streamlit session_state 初始化
def init_state():
    if "workspace" not in st.session_state:
        st.session_state.workspace = Workspace.load()
    if "default_model" not in st.session_state:
        st.session_state.default_model = ModelRegistry.get_default()
```

---

## 9. AI Agent 接口设计（v1.0+）

### 9.1 设计理念

AI Agent 通过结构化的工具函数与 VisionWorkbench 交互。所有工具函数：
- 接受明确的类型化参数
- 返回 JSON 可序列化的结果
- 包含人类可读和 AI 可读的输出

### 9.2 Agent 工具定义

```python
# agent/tools.py

AGENT_TOOLS = [
    {
        "name": "list_datasets",
        "description": "列出所有已注册的数据集",
        "parameters": {},
        "returns": "list[DatasetInfo]"
    },
    {
        "name": "validate_dataset",
        "description": "校验指定数据集",
        "parameters": {"dataset_id": "int"},
        "returns": "ValidationReport"
    },
    {
        "name": "list_models",
        "description": "列出所有已注册的模型",
        "parameters": {},
        "returns": "list[ModelInfo]"
    },
    {
        "name": "compare_models",
        "description": "比较多个模型的指标",
        "parameters": {"model_version_ids": "list[int]"},
        "returns": "ComparisonResult"
    },
    {
        "name": "analyze_errors",
        "description": "分析评估中发现的错误模式",
        "parameters": {"evaluation_id": "int"},
        "returns": "ErrorAnalysis"
    },
    # ... 更多工具
]
```

### 9.3 Agent 交互模式

```
用户: "为什么我的模型对小目标检测效果不好？"

AI Agent:
  1. 调用 list_models() 获取模型列表
  2. 调用 get_evaluations(model_id) 获取评估结果
  3. 调用 analyze_errors(eval_id) 获取错误分析
  4. 调用 get_dataset_statistics(dataset_id) 获取数据分布
  5. 综合信息给出分析

AI Agent 回复: "你的模型 v3 在 COCO 小目标类别上的 mAP 仅 0.23，
  远低于大目标的 0.78。同时你的数据集中小目标样本仅占 8%。
  建议：1) 增加小目标样本 2) 降低输入分辨率 3) 尝试 mosaic 增强..."
```

---

## 10. 训练子进程架构

### 10.1 进程模型

```python
import subprocess
import threading

class TrainingLauncher:
    def start(self, task_id: int) -> None:
        task = self._get_task(task_id)
        config = self._build_config(task)

        # 在子进程中启动 YOLO 训练
        process = subprocess.Popen(
            ["python", "-m", "vw.train_worker",
             "--config", config.config_path,
             "--task-id", str(task_id)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        # 更新任务状态
        task.status = "running"
        task.pid = process.pid
        self._save_task(task)

        # 后台线程收集日志
        threading.Thread(
            target=self._collect_logs,
            args=(task_id, process),
            daemon=True
        ).start()
```

### 10.2 训练 Worker

```python
# train_worker.py — 在子进程中运行
from ultralytics import YOLO

def main(config_path: str, task_id: int):
    config = load_config(config_path)
    model = YOLO(config.base_model)

    # Ultralytics 训练
    results = model.train(
        data=config.data_yaml,
        epochs=config.epochs,
        batch=config.batch,
        imgsz=config.imgsz,
        device=config.device,
        project=config.project_dir,
        name=f"task_{task_id}",
        exist_ok=True
    )

    # 训练完成后写完成标记
    write_completion_marker(task_id, results)
```

---

## 11. 配置管理

### 11.1 配置文件结构

```yaml
# ~/visionworkbench/config.yaml（自动生成）
workspace:
  name: "my_workspace"
  path: "~/visionworkbench"

defaults:
  model: null          # 默认模型 ID
  device: "cpu"        # cpu, cuda:0
  confidence: 0.5      # 默认置信度阈值

training:
  default_epochs: 100
  default_batch: 16
  default_imgsz: 640
  default_device: "cpu"

ui:
  theme: "light"
  language: "zh"
  sidebar_expanded: true
```

### 11.2 配置优先级

```
命令行参数 > 环境变量 > 工作区 config.yaml > 系统 defaults.yaml
```

---

## 12. 版本演进对架构的影响

| 版本 | 新增模块 | 架构变更 |
|------|---------|---------|
| v0.1 | core/model/inference | 最小架构：main.py + inference.py |
| v0.2 | core/dataset/* | 引入 infrastructure/database.py + filesystem.py |
| v0.3 | core/training/* | 引入子进程管理 |
| v0.4 | core/model/registry + lineage | 引入模型血统追踪 |
| v0.5 | core/experiment/* | 引入实验对比逻辑 |
| v1.0 | core/evaluation/*, app/demo_page | 全模块上线，引入 app/ 页面层 |
| v1.x | agent/* | 引入 AI Agent 工具层 |

---

## 13. 安全与隐私

| 维度 | 措施 |
|------|------|
| 数据隐私 | 不联网，不上传任何数据 |
| 代码执行 | 子进程训练，进程隔离 |
| 文件安全 | 仅在工作区目录内读写 |
| 依赖安全 | uv lock 锁定依赖版本 |
| 配置文件 | 不存储密码或密钥 |
| 模型文件 | 只读加载，不修改原始模型文件 |

---

## 14. 测试策略

```
tests/
├── unit/
│   ├── test_dataset_registry.py
│   ├── test_dataset_validator.py
│   ├── test_model_registry.py
│   ├── test_training_launcher.py
│   ├── test_experiment_manager.py
│   └── test_evaluation_runner.py
├── integration/
│   ├── test_dataset_to_training.py
│   ├── test_training_to_model.py
│   └── test_full_workflow.py
└── fixtures/
    ├── sample_dataset/      # 小样本 YOLO 数据集
    └── sample_model.pt      # 小模型文件
```

测试框架：pytest
CI 策略：GitHub Actions（可选，本地开发为主）

---

## 15. 关键设计决策记录

| 决策 | 选项 A | 选项 B | 选择 | 理由 |
|------|--------|--------|------|------|
| UI 框架 | Streamlit | Gradio | **Streamlit** | 更适合多页面应用，社区更大 |
| 数据库 | SQLite | PostgreSQL | **SQLite** | 零配置，单文件，足够用 |
| 进程模型 | 线程 | 子进程 | **子进程** | 训练需要进程隔离，OOM 不影响主进程 |
| 配置文件 | YAML | TOML | **YAML** | 学生更熟悉，AI 更易生成 |
| 包管理 | pip | uv | **uv** | 更快，有锁文件 |

---

## 16. 启动流程

```
用户执行: vw start
    │
    ▼
CLI 入口 (cli.py)
    ├── 检查 Python 版本 (3.10+)
    ├── 检查/创建工作区目录
    ├── 初始化 SQLite 数据库 (运行迁移)
    ├── 加载配置文件
    └── 启动 Streamlit
         │
         ▼
    streamlit run vw/main.py
         │
         ▼
    浏览器自动打开 http://localhost:8501
```

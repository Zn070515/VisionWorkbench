# VisionWorkbench

轻量级本地优先 AI 视觉研发工作台 — 面向学生、创客和竞赛团队的完整 CV 实验环境。

## 功能

- **概览** — 单模型图片/摄像头推理，支持 YOLO .pt 模型加载与可视化
- **数据集** — 注册、校验、统计 YOLO 格式数据集
- **训练** — 子进程 YOLO 训练，PID 追踪，实时日志与指标图表
- **实验** — 多模型对比实验，指标聚合与结论记录
- **模型** — 模型注册表，版本管理与血统追踪
- **评估** — 批量图片/视频评估，逐类误差分析
- **演示** — 视频/摄像头实时推理演示

## 快速开始

```bash
# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate # Linux/macOS

# 安装依赖
pip install -e .
# 如需 CUDA 加速：
pip install -e ".[cuda]"

# 启动应用
streamlit run vw/main.py
```

## 技术栈

| 层 | 选型 |
|---|---|
| 界面 | Streamlit |
| 视觉模型 | Ultralytics YOLO |
| 图像处理 | OpenCV, Pillow |
| 数据存储 | SQLite |
| 可视化 | Plotly, Pandas |
| 平台 | Windows 10/11 优先，兼容 Linux |

## 架构

```
vw/
├── app/          # Streamlit 页面
├── core/         # 业务逻辑
│   ├── dataset/     # 数据集管理
│   ├── training/    # 训练管理器
│   ├── model/       # 模型注册表
│   ├── experiment/  # 实验中心
│   └── evaluation/  # 评估执行器
├── infrastructure/  # SQLite 数据库 & 文件系统
└── utils/           # 图像/文件工具
```

## 开发

```bash
# 运行测试
pip install -e ".[dev]"
pytest tests/ -v
```

## 许可

MIT

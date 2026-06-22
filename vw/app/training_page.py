"""v0.3 训练页面 — 配置、启动、监控训练任务."""

import time
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from vw.core.dataset.registry import DatasetRegistry
from vw.core.training.manager import TrainingManager


YOLO_MODELS = [
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
    "yolov8n-cls.pt", "yolov8s-cls.pt", "yolov8m-cls.pt",
    "yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt",
    "yolov8n-pose.pt", "yolov8s-pose.pt",
    "yolov9t.pt", "yolov9s.pt", "yolov9m.pt", "yolov9c.pt", "yolov9e.pt",
    "yolov10n.pt", "yolov10s.pt", "yolov10m.pt", "yolov10b.pt",
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
    "yolo12n.pt", "yolo12s.pt", "yolo12m.pt", "yolo12l.pt", "yolo12x.pt",
]


def init_session():
    if "tm" not in st.session_state:
        st.session_state.tm = TrainingManager()
    if "reg" not in st.session_state:
        st.session_state.reg = DatasetRegistry()
    if "train_log_auto_refresh" not in st.session_state:
        st.session_state.train_log_auto_refresh = False


def render_sidebar():
    """渲染侧边栏：训练配置 + 任务列表."""
    st.sidebar.title("VisionWorkbench v0.3")
    st.sidebar.markdown("---")

    _render_config_form()
    st.sidebar.markdown("---")
    _render_task_list()


def _render_config_form():
    """训练配置表单."""
    st.sidebar.subheader("训练配置")

    reg: DatasetRegistry = st.session_state.reg
    datasets = reg.list_all()
    ds_options = {f"{d['name']} (ID:{d['id']})": d for d in datasets}
    ds_choice = st.sidebar.selectbox(
        "数据集", list(ds_options.keys()),
        key="train_ds",
        disabled=not ds_options,
    )

    base_model = st.sidebar.selectbox(
        "基础模型", YOLO_MODELS,
        index=YOLO_MODELS.index("yolov8n.pt"),
        key="train_base_model",
    )
    model_name = st.sidebar.text_input("任务名称", "my_train", key="train_name")

    col1, col2 = st.sidebar.columns(2)
    epochs = col1.number_input("Epochs", 1, 1000, 100, 10, key="train_epochs")
    batch = col2.number_input("Batch", 1, 256, 16, 4, key="train_batch")

    col1, col2 = st.sidebar.columns(2)
    imgsz = col1.number_input("Imgsz", 320, 1920, 640, 32, key="train_imgsz")
    device = col2.selectbox("Device", ["cpu", "0"], key="train_device")

    if not ds_options:
        st.sidebar.warning("请先在数据集页面注册数据集")

    tm: TrainingManager = st.session_state.tm
    running = any(t["status"] == "running" for t in tm.list_tasks())

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("启动训练", type="primary", use_container_width=True,
                     disabled=running or not ds_options):
            ds = ds_options[ds_choice]
            data_yaml = Path(ds["path"]) / "data.yaml"
            if not data_yaml.exists():
                st.sidebar.error(f"数据集缺少 data.yaml: {data_yaml}")
            else:
                config = {
                    "data": str(data_yaml),
                    "epochs": epochs,
                    "imgsz": imgsz,
                    "batch": batch,
                    "device": device,
                }
                task_id = tm.create_task(
                    model_name=model_name,
                    base_model=base_model,
                    config=config,
                    dataset_id=ds["id"],
                )
                tm.launch(task_id)
                st.session_state.selected_task_id = task_id
                st.session_state.train_log_auto_refresh = True
                st.rerun()

    with col2:
        if st.button("停止训练", use_container_width=True, disabled=not running):
            for t in tm.list_tasks():
                if t["status"] == "running":
                    tm.stop(t["id"])
            st.session_state.train_log_auto_refresh = False
            st.rerun()


def _render_task_list():
    """侧边栏任务列表."""
    st.sidebar.subheader("训练历史")
    tm: TrainingManager = st.session_state.tm
    tasks = tm.list_tasks()

    if not tasks:
        st.sidebar.info("暂无训练任务")
        return

    status_labels = {
        "pending": "⏳", "running": "🔄", "completed": "✓",
        "failed": "✗", "stopped": "⏹",
    }

    for t in tasks:
        label = status_labels.get(t["status"], "?")
        if st.sidebar.button(
            f"{label} {t['model_name']} ({t['status']})",
            key=f"task_{t['id']}",
            use_container_width=True,
        ):
            st.session_state.selected_task_id = t["id"]
            st.session_state.train_log_auto_refresh = (t["status"] == "running")
            st.rerun()


def render_main():
    """渲染主区域：训练监控."""
    st.title("训练中心")
    st.caption("配置、启动、监控 YOLO 训练任务")

    if "selected_task_id" not in st.session_state:
        st.info("👈 配置训练参数并启动，或选择历史任务")
        return

    tm: TrainingManager = st.session_state.tm
    task = tm.get_task(st.session_state.selected_task_id)
    if not task:
        st.error("任务不存在")
        return

    _render_status_bar(task)
    st.markdown("---")

    tab1, tab2 = st.tabs(["指标 & 日志", "任务详情"])

    with tab1:
        _render_metrics(task)
        _render_log(task)

    with tab2:
        _render_task_detail(task)

    # 自动刷新
    if task["status"] == "running":
        tm.poll_completion(task["id"])
        if st.session_state.train_log_auto_refresh:
            time.sleep(2)
            st.rerun()


def _render_status_bar(task: dict):
    """渲染状态栏."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("任务", task["model_name"])
    col2.metric("基础模型", task["base_model"])
    status_display = {"running": "🔄 训练中", "completed": "✓ 完成",
                      "failed": "✗ 失败", "stopped": "⏹ 已停止",
                      "pending": "⏳ 等待中"}
    col3.metric("状态", status_display.get(task["status"], task["status"]))
    col4.metric("创建时间", (task.get("created_at") or "")[:16])

    if task["status"] == "running":
        st.progress(0.0, text="训练进行中...")


def _render_metrics(task: dict):
    """渲染指标图表."""
    tm: TrainingManager = st.session_state.tm
    metrics = tm.get_metrics(task["id"])

    if not metrics:
        if task["status"] == "running":
            st.info("等待指标数据...")
        else:
            st.info("暂无指标数据")
        return

    epochs = [m["epoch"] for m in metrics]

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=[m["box_loss"] for m in metrics],
                                 mode="lines+markers", name="Box Loss"))
        fig.add_trace(go.Scatter(x=epochs, y=[m["cls_loss"] for m in metrics],
                                 mode="lines+markers", name="Cls Loss"))
        fig.add_trace(go.Scatter(x=epochs, y=[m["dfl_loss"] for m in metrics],
                                 mode="lines+markers", name="DFL Loss"))
        fig.update_layout(title="Loss 曲线", height=350,
                          xaxis_title="Epoch", yaxis_title="Loss")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        if metrics and "map50" in metrics[0]:
            fig.add_trace(go.Scatter(x=epochs, y=[m["map50"] for m in metrics],
                                     mode="lines+markers", name="mAP50"))
        if metrics and "map50_95" in metrics[0]:
            fig.add_trace(go.Scatter(x=epochs, y=[m["map50_95"] for m in metrics],
                                     mode="lines+markers", name="mAP50-95"))
        if metrics and "precision" in metrics[0]:
            fig.add_trace(go.Scatter(x=epochs, y=[m["precision"] for m in metrics],
                                     mode="lines+markers", name="Precision"))
        if metrics and "recall" in metrics[0]:
            fig.add_trace(go.Scatter(x=epochs, y=[m["recall"] for m in metrics],
                                     mode="lines+markers", name="Recall"))
        fig.update_layout(title="mAP & 指标", height=350,
                          xaxis_title="Epoch", yaxis_title="值")
        st.plotly_chart(fig, use_container_width=True)

    # 指标数据表格
    if metrics:
        with st.expander(f"指标明细 ({len(metrics)} epochs)"):
            df = pd.DataFrame(metrics)
            st.dataframe(df, use_container_width=True, hide_index=True)


def _render_log(task: dict):
    """渲染日志查看器."""
    st.subheader("训练日志")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        tail_lines = st.slider("显示行数", 20, 500, 80, 20, key="log_tail")
    with col2:
        auto = st.checkbox("自动刷新", key="train_log_auto_refresh_widget")
        if auto != st.session_state.train_log_auto_refresh:
            st.session_state.train_log_auto_refresh = auto

    tm: TrainingManager = st.session_state.tm
    log_text = tm.get_log(task["id"], tail=tail_lines)

    if log_text:
        st.code(log_text, language="text", line_numbers=False)
    else:
        st.info("暂无日志输出")


def _render_task_detail(task: dict):
    """渲染任务详情."""
    import json

    st.subheader("配置参数")
    try:
        config = json.loads(task["config"]) if isinstance(task["config"], str) else task["config"]
        st.json(config)
    except Exception:
        st.text(str(task.get("config", "")))

    st.subheader("文件路径")
    col1, col2 = st.columns(2)
    with col1:
        st.text(f"日志: {task.get('log_path', 'N/A')}")
        st.text(f"结果: {task.get('results_path', 'N/A')}")
    with col2:
        st.text(f"开始: {task.get('started_at', 'N/A')}")
        st.text(f"结束: {task.get('finished_at', 'N/A')}")

    st.markdown("---")
    if st.button("删除任务", type="secondary"):
        tm: TrainingManager = st.session_state.tm
        tm.delete_task(task["id"])
        del st.session_state.selected_task_id
        st.session_state.train_log_auto_refresh = False
        st.rerun()


def render():
    init_session()
    render_sidebar()
    render_main()

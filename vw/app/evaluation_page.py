"""v1.0 评估页面 — 批量推理、视频评估、错误分析."""

import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from vw.core.model.registry import ModelRegistry
from vw.core.dataset.registry import DatasetRegistry
from vw.core.evaluation.runner import EvaluationRunner


def init_session():
    if "eval_runner" not in st.session_state:
        st.session_state.eval_runner = EvaluationRunner()
    if "eval_model_reg" not in st.session_state:
        st.session_state.eval_model_reg = ModelRegistry()
    if "eval_ds_reg" not in st.session_state:
        st.session_state.eval_ds_reg = DatasetRegistry()


def render_sidebar():
    st.sidebar.markdown("---")
    st.sidebar.subheader("评估配置")

    # 评估模式
    mode = st.sidebar.radio("模式", ["图片批评估", "视频评估"], key="eval_mode")
    st.session_state.eval_mode = mode

    # 模型选择
    model_reg: ModelRegistry = st.session_state.eval_model_reg
    models = model_reg.list_all()
    if models:
        model_opts = {}
        for m in models:
            versions = model_reg.get_versions(m["id"])
            for v in versions:
                label = f"{m['name']} v{v['version']}"
                model_opts[label] = (v["file_path"], v["id"])
        model_choice = st.sidebar.selectbox("模型", list(model_opts.keys()), key="eval_model")
        st.session_state.eval_model_path = model_opts[model_choice][0]
        st.session_state.eval_model_version_id = model_opts[model_choice][1]
    else:
        st.sidebar.warning("请先在模型页面注册模型")
        st.session_state.eval_model_path = None

    confidence = st.sidebar.slider("置信度", 0.0, 1.0, 0.5, 0.05, key="eval_conf")
    iou_thresh = st.sidebar.slider("IoU 阈值", 0.1, 0.9, 0.5, 0.05, key="eval_iou")
    st.session_state.eval_confidence = confidence
    st.session_state.eval_iou_threshold = iou_thresh

    if mode == "图片批评估":
        ds_reg: DatasetRegistry = st.session_state.eval_ds_reg
        datasets = ds_reg.list_all()
        if datasets:
            ds_opts = {d["name"]: d for d in datasets}
            ds_choice = st.sidebar.selectbox("数据集", list(ds_opts.keys()), key="eval_ds")
            st.session_state.eval_dataset = ds_opts[ds_choice]
        else:
            st.sidebar.warning("请先注册数据集")

        max_img = st.sidebar.number_input("最大图片数 (0=全部)", 0, 10000, 100, key="eval_max")
        st.session_state.eval_max_images = max_img if max_img > 0 else None

        if st.sidebar.button("运行评估", type="primary", use_container_width=True,
                             disabled=not (st.session_state.get("eval_model_path")
                                           and st.session_state.get("eval_dataset"))):
            with st.spinner("评估中..."):
                runner: EvaluationRunner = st.session_state.eval_runner
                ds = st.session_state.eval_dataset
                results = runner.run_on_dataset(
                    model_path=st.session_state.eval_model_path,
                    dataset_path=ds["path"],
                    confidence=confidence,
                    iou_threshold=iou_thresh,
                    max_images=st.session_state.eval_max_images,
                )
                st.session_state.eval_results = results
                st.session_state.eval_error_analysis = runner.error_analysis(results)
                if results.get("valid"):
                    runner.save(
                        model_version_id=st.session_state.eval_model_version_id,
                        eval_type="image",
                        input_path=ds["path"],
                        results=results,
                        dataset_id=ds["id"],
                    )
            st.rerun()

    else:
        video_file = st.sidebar.file_uploader("视频文件", type=["mp4", "avi", "mov", "mkv"], key="eval_video")
        sample = st.sidebar.number_input("采样间隔 (帧)", 1, 100, 1, key="eval_sample")
        if video_file:
            with tempfile.NamedTemporaryFile(suffix=Path(video_file.name).suffix, delete=False) as tmp:
                tmp.write(video_file.read())
                st.session_state.eval_video_path = tmp.name

        if st.sidebar.button("运行视频评估", type="primary", use_container_width=True,
                             disabled=not (st.session_state.get("eval_model_path")
                                           and st.session_state.get("eval_video_path"))):
            with st.spinner("视频评估中..."):
                runner: EvaluationRunner = st.session_state.eval_runner
                results = runner.run_on_video(
                    model_path=st.session_state.eval_model_path,
                    video_path=st.session_state.eval_video_path,
                    confidence=confidence,
                    sample_every=sample,
                )
                st.session_state.eval_results = results
                if results.get("valid"):
                    runner.save(
                        model_version_id=st.session_state.eval_model_version_id,
                        eval_type="video",
                        input_path=st.session_state.eval_video_path,
                        results=results,
                    )
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("评估历史")
    runner: EvaluationRunner = st.session_state.eval_runner
    history = runner.get_history(limit=20)
    if history:
        for h in history:
            label = f"{h['type']} — {h.get('created_at', '')[:16]}"
            if st.sidebar.button(label, key=f"eval_h_{h['id']}", use_container_width=True):
                st.session_state.eval_results = json_loads(h["results"])
                st.rerun()
    else:
        st.sidebar.caption("暂无评估记录")


def render_main():
    st.title("评估中心")
    st.caption("批量推理、视频评估、错误分析")

    if "eval_results" not in st.session_state:
        st.info("👈 配置评估参数并运行")
        return

    results = st.session_state.eval_results

    if not results.get("valid"):
        st.error(results.get("error", "评估失败"))
        return

    if "frames" in results:
        _render_video_results(results)
    else:
        _render_image_results(results)

    if "eval_error_analysis" in st.session_state:
        _render_error_analysis(st.session_state.eval_error_analysis)


def _render_image_results(results: dict):
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("图片数", results.get("total_images", 0))
    col2.metric("真值框", results.get("total_gt_boxes", 0))
    col3.metric("预测框", results.get("total_pred_boxes", 0))
    col4.metric("Precision", f"{results.get('precision', 0):.4f}")
    col5.metric("Recall", f"{results.get('recall', 0):.4f}")

    st.markdown("---")

    # 每类指标
    per_class = results.get("per_class", {})
    if per_class:
        st.subheader("各类别指标")
        rows = []
        for cls_id, counts in sorted(per_class.items()):
            rows.append({
                "类别": counts.get("class_name", cls_id),
                "TP": counts["tp"],
                "FP": counts["fp"],
                "FN": counts["fn"],
                "Precision": f"{counts.get('precision', 0):.4f}",
                "Recall": f"{counts.get('recall', 0):.4f}",
                "F1": f"{counts.get('f1', 0):.4f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # 柱状图
        names = [r["类别"] for r in rows]
        prec_vals = [float(r["Precision"]) for r in rows]
        rec_vals = [float(r["Recall"]) for r in rows]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Precision", x=names, y=prec_vals))
        fig.add_trace(go.Bar(name="Recall", x=names, y=rec_vals))
        fig.update_layout(height=350, barmode="group",
                          xaxis_title="类别", yaxis_title="值")
        st.plotly_chart(fig, use_container_width=True)


def _render_video_results(results: dict):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总帧数", results.get("total_frames", 0))
    col2.metric("采样帧", results.get("sampled_frames", 0))
    col3.metric("FPS", f"{results.get('fps', 0):.1f}")
    col4.metric("总检测数", results.get("total_detections", 0))

    # 时间序列检测计数
    frames = results.get("frames", [])
    if frames:
        st.subheader("检测时序")
        timestamps = [f["timestamp"] for f in frames]
        counts = [f["detection_count"] for f in frames]
        fig = px.line(x=timestamps, y=counts, labels={"x": "时间 (秒)", "y": "检测数"})
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # 帧列表
        with st.expander(f"采样帧详情 ({len(frames)} 帧)"):
            rows = []
            for f in frames:
                rows.append({
                    "帧": f["frame_idx"],
                    "时间": f"{f['timestamp']:.1f}s",
                    "检测数": f["detection_count"],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_error_analysis(analysis: dict):
    st.markdown("---")
    st.subheader("错误分析")

    worst = analysis.get("worst_classes", [])
    if worst:
        st.write("**表现最差类别:**")
        for w in worst:
            st.text(
                f"  {w['class_name']}: F1={w['f1']:.3f} "
                f"P={w['precision']:.3f} R={w['recall']:.3f} "
                f"(FP={w['fp']} FN={w['fn']})"
            )

    conf = analysis.get("confidence_distribution", {})
    if any(conf.values()):
        fig = px.pie(
            names=["高置信度 (>0.7)", "中置信度 (0.3-0.7)", "低置信度 (<0.3)"],
            values=[conf["high"], conf["medium"], conf["low"]],
            title="误检置信度分布",
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


def json_loads(s):
    import json
    try:
        return json.loads(s) if isinstance(s, str) else s
    except Exception:
        return {}


def render():
    init_session()
    render_sidebar()
    render_main()

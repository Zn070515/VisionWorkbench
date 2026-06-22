"""v0.4 模型页面 — 注册、版本、血统追踪."""

import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from vw.core.model.registry import ModelRegistry


def init_session():
    if "model_registry" not in st.session_state:
        st.session_state.model_registry = ModelRegistry()


def render_sidebar():
    st.sidebar.markdown("---")
    st.sidebar.subheader("注册模型")

    name = st.sidebar.text_input("模型名称", key="reg_model_name")
    model_file = st.sidebar.file_uploader(
        "模型文件 (.pt)", type=["pt"], key="reg_model_file"
    )
    task = st.sidebar.selectbox(
        "任务类型", ["detect", "segment", "classify", "pose", "obb"],
        key="reg_model_task",
    )
    imgsz = st.sidebar.number_input("输入尺寸", 160, 1920, 640, 32, key="reg_model_imgsz")
    num_cls = st.sidebar.number_input("类别数", 0, 10000, 0, 1, key="reg_model_nc")

    if st.sidebar.button("注册模型", use_container_width=True, disabled=not (name and model_file)):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp.write(model_file.read())
            tmp_path = tmp.name
        reg: ModelRegistry = st.session_state.model_registry
        reg.register(
            name=name,
            file_path=tmp_path,
            task_type=task,
            num_classes=num_cls if num_cls > 0 else None,
            input_size=imgsz,
        )
        st.sidebar.success(f"已注册: {name}")
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("已注册模型")

    reg: ModelRegistry = st.session_state.model_registry
    models = reg.list_all()

    if models:
        for m in models:
            default_badge = " ★" if m["is_default"] else ""
            versions = reg.get_versions(m["id"])
            ver_count = len(versions)
            latest_map = f"{versions[0]['map50']:.3f}" if versions and versions[0].get("map50") else "-"

            label = f"{m['name']}{default_badge}  v{ver_count}  mAP:{latest_map}"
            if st.sidebar.button(label, key=f"model_{m['id']}", use_container_width=True):
                st.session_state.selected_model_id = m["id"]
                st.rerun()
    else:
        st.sidebar.info("暂无模型")


def render_main():
    st.title("模型注册表")
    st.caption("管理模型版本、指标和血统")

    if "selected_model_id" not in st.session_state:
        st.info("👈 注册新模型或选择已有模型")
        return

    reg: ModelRegistry = st.session_state.model_registry
    model = reg.get(st.session_state.selected_model_id)
    if not model:
        st.error("模型不存在")
        return

    # ── 基本信息 ──
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("名称", model["name"])
    col2.metric("类型", model["type"])
    col3.metric("任务", model.get("task_type") or "-")
    col4.metric("类别数", str(model.get("num_classes") or "-"))
    col5.metric("默认", "★" if model["is_default"] else "-")

    st.markdown("---")

    # ── 版本列表 ──
    versions = reg.get_versions(model["id"])
    if versions:
        st.subheader(f"版本历史 ({len(versions)})")

        rows = []
        for v in versions:
            rows.append({
                "版本": f"v{v['version']}",
                "mAP50": f"{v['map50']:.4f}" if v.get("map50") else "-",
                "mAP50-95": f"{v['map50_95']:.4f}" if v.get("map50_95") else "-",
                "Precision": f"{v['precision']:.4f}" if v.get("precision") else "-",
                "Recall": f"{v['recall']:.4f}" if v.get("recall") else "-",
                "大小": f"{v['file_size'] / 1e6:.1f} MB" if v.get("file_size") else "-",
                "创建时间": (v.get("created_at") or "")[:16],
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # 指标对比图
        if len(versions) > 1 and any(v.get("map50") for v in versions):
            st.subheader("版本指标趋势")
            vers = sorted(versions, key=lambda v: v["version"])
            labels = [f"v{v['version']}" for v in vers]

            fig = go.Figure()
            for metric, mname in [("map50", "mAP50"), ("map50_95", "mAP50-95"),
                                   ("precision", "Precision"), ("recall", "Recall")]:
                vals = [v.get(metric) for v in vers if v.get(metric) is not None]
                if len(vals) == len(vers):
                    fig.add_trace(go.Scatter(
                        x=labels, y=vals, mode="lines+markers", name=mname,
                    ))
            fig.update_layout(height=350, xaxis_title="版本", yaxis_title="值")
            st.plotly_chart(fig, use_container_width=True)

    # ── 血统 ──
    st.markdown("---")
    st.subheader("模型血统")

    lineage = reg.get_lineage(model["id"])
    if lineage.get("versions"):
        for v_info in lineage["versions"]:
            parts = [f"**v{v_info['version']}**"]
            if v_info.get("training_task"):
                t = v_info["training_task"]
                parts.append(f"← 训练: {t['model_name']} ({t['status']})")
                if v_info.get("dataset"):
                    d = v_info["dataset"]
                    parts.append(f"← 数据集: {d['name']} (ID:{d['id']})")
            st.markdown("  " + "  ".join(parts))
            if v_info.get("notes"):
                st.caption(f"  备注: {v_info['notes']}")
    else:
        st.info("暂无血统信息")

    # ── 操作 ──
    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("设为默认", use_container_width=True,
                     disabled=bool(model["is_default"])):
            reg.set_default(model["id"])
            st.rerun()
    with col_b:
        if st.button("导出", use_container_width=True):
            dest = reg.workspace / "exports" / model["name"]
            reg.export(model["id"], dest)
            st.success(f"已导出到 {dest}")
    with col_c:
        if st.button("删除模型", type="secondary", use_container_width=True):
            reg.delete(model["id"])
            del st.session_state.selected_model_id
            st.rerun()

    # ── 手动添加版本 ──
    st.markdown("---")
    with st.expander("添加新版本"):
        ver_file = st.file_uploader("模型文件", type=["pt"], key="ver_file")
        ver_notes = st.text_input("版本备注", key="ver_notes")
        if st.button("添加版本", disabled=not ver_file):
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                tmp.write(ver_file.read())
                tmp_path = tmp.name
            reg.add_version(model["id"], tmp_path, notes=ver_notes or None)
            st.success(f"已添加新版本")
            st.rerun()


def render():
    init_session()
    render_sidebar()
    render_main()

"""v0.2 数据集页面 — 注册、校验、统计."""

from pathlib import Path

import streamlit as st
import pandas as pd

from vw.core.dataset.registry import DatasetRegistry
from vw.core.dataset.validator import validate_dataset
from vw.core.dataset.statistics import get_statistics


def init_session():
    if "registry" not in st.session_state:
        st.session_state.registry = DatasetRegistry()
    if "validation_result" not in st.session_state:
        st.session_state.validation_result = None


def render_sidebar():
    st.sidebar.title("VisionWorkbench v0.2")
    st.sidebar.markdown("---")
    st.sidebar.subheader("注册新数据集")

    name = st.sidebar.text_input("数据集名称", key="reg_name")
    ds_path = st.sidebar.text_input("数据集路径", key="reg_path")
    ds_type = st.sidebar.selectbox("格式", ["yolo"], key="reg_type")

    if st.sidebar.button("注册", use_container_width=True, disabled=not (name and ds_path)):
        reg: DatasetRegistry = st.session_state.registry
        p = Path(ds_path)
        if not p.exists():
            st.sidebar.error("路径不存在")
        elif reg.get_by_path(ds_path):
            st.sidebar.warning("该路径已注册")
        else:
            reg.register(name, ds_path, ds_type)
            st.sidebar.success(f"已注册: {name}")
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("已注册数据集")

    datasets = st.session_state.registry.list_all()
    if datasets:
        for ds in datasets:
            status_icon = {"validated": "✓", "error": "✗", "registered": "○"}.get(ds["status"], "?")
            if st.sidebar.button(
                f"{status_icon} {ds['name']}",
                key=f"ds_{ds['id']}",
                use_container_width=True,
            ):
                st.session_state.selected_dataset_id = ds["id"]
                st.session_state.validation_result = None
                st.rerun()
    else:
        st.sidebar.info("暂无数据集")


def render_main():
    st.title("数据集中心")
    st.caption("注册、校验、统计分析数据集")

    if "selected_dataset_id" not in st.session_state:
        st.info("👈 注册或选择数据集")
        return

    reg: DatasetRegistry = st.session_state.registry
    ds = reg.get(st.session_state.selected_dataset_id)
    if not ds:
        st.error("数据集不存在")
        return

    # 数据集信息
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("名称", ds["name"])
    col2.metric("格式", ds["type"])
    col3.metric("状态", ds["status"])
    col4.metric("路径", str(Path(ds["path"]).name))

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("运行校验", type="primary", use_container_width=True):
            with st.spinner("校验中..."):
                result = validate_dataset(ds["path"])
                st.session_state.validation_result = result
                # 更新数据库状态
                status = "validated" if result.valid else "error"
                reg.update_status(
                    ds["id"], status,
                    num_images=result.stats.get("total_images", 0),
                    num_classes=result.stats.get("num_classes", 0),
                )
            st.rerun()

    with col_b:
        if st.button("查看统计", use_container_width=True):
            with st.spinner("统计中..."):
                st.session_state.show_statistics = not st.session_state.get("show_statistics", False)
            st.rerun()

    # 校验结果
    result = st.session_state.validation_result
    if result:
        st.markdown("---")
        st.subheader("校验报告")

        d = result.to_dict()
        col1, col2, col3 = st.columns(3)
        col1.metric("校验通过", "✓" if d["valid"] else "✗")
        col2.metric("错误", str(len(d["errors"])))
        col3.metric("警告", str(len(d["warnings"])))

        if d["errors"]:
            st.error("\n".join(f"- {e}" for e in d["errors"]))
        if d["warnings"]:
            st.warning("\n".join(f"- {w}" for w in d["warnings"]))
        if d["info"]:
            with st.expander("详细信息", expanded=True):
                for info in d["info"]:
                    st.text(info)

        # 统计图表
        stats = d.get("stats", {})
        if stats.get("class_distribution"):
            st.subheader("类别分布")
            dist = stats["class_distribution"]
            import plotly.express as px
            fig = px.bar(
                x=[f"类别 {k}" for k in sorted(dist.keys())],
                y=[dist[k] for k in sorted(dist.keys())],
                labels={"x": "类别", "y": "标注框数量"},
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        if stats.get("bbox_size_distribution"):
            st.subheader("目标尺寸分布")
            bbox_dist = stats["bbox_size_distribution"]
            fig = px.pie(
                names=list(bbox_dist.keys()),
                values=list(bbox_dist.values()),
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    # 详细统计
    if st.session_state.get("show_statistics"):
        st.markdown("---")
        st.subheader("详细统计")
        with st.spinner("计算中..."):
            full_stats = get_statistics(ds["path"])
        col1, col2 = st.columns(2)
        with col1:
            if "image_size" in full_stats:
                st.write("**图片尺寸**")
                img_sz = full_stats["image_size"]
                st.text(
                    f"  平均: {img_sz['avg_width']}x{img_sz['avg_height']}\n"
                    f"  范围: {img_sz['min_width']}x{img_sz['min_height']}"
                    f" ~ {img_sz['max_width']}x{img_sz['max_height']}"
                )
        with col2:
            if "bbox_stats" in full_stats:
                st.write("**标注框尺寸（归一化）**")
                bb = full_stats["bbox_stats"]
                st.text(
                    f"  平均: {bb['avg_width']:.3f} x {bb['avg_height']:.3f}\n"
                    f"  范围: {bb['min_width']:.3f} ~ {bb['max_width']:.3f}"
                )

    # 删除按钮
    st.markdown("---")
    if st.button("删除数据集", type="secondary"):
        reg.delete(ds["id"])
        del st.session_state.selected_dataset_id
        st.session_state.validation_result = None
        st.rerun()


def render():
    init_session()
    render_sidebar()
    render_main()

"""VisionWorkbench Streamlit 入口."""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 上，兼容 python vw/main.py 和 streamlit run vw/main.py
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st

from vw.app import home, dataset_page, training_page, model_page, experiment_page, evaluation_page, demo_page

PAGES = {
    "概览": home,
    "数据集": dataset_page,
    "训练": training_page,
    "实验": experiment_page,
    "模型": model_page,
    "评估": evaluation_page,
    "演示": demo_page,
}


def main():
    st.set_page_config(
        page_title="VisionWorkbench",
        page_icon="🔬",
        layout="wide",
    )

    st.sidebar.title("VisionWorkbench")
    st.sidebar.caption("v1.0 — 完整工作台")

    page = st.sidebar.radio(
        "导航",
        list(PAGES.keys()),
        format_func=lambda x: f"{'  ' if PAGES[x] is None else ''}{x}{' (开发中)' if PAGES[x] is None else ''}",
    )

    st.sidebar.markdown("---")

    module = PAGES[page]
    if module:
        module.render()
    else:
        st.title(page)
        st.info("该模块正在开发中，敬请期待。")


if __name__ == "__main__":
    main()

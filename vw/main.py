"""VisionWorkbench Streamlit 入口."""

import streamlit as st

from vw.app import home, dataset_page

PAGES = {
    "概览": home,
    "数据集": dataset_page,
    "训练": None,
    "实验": None,
    "模型": None,
    "评估": None,
    "演示": None,
}


def main():
    st.set_page_config(
        page_title="VisionWorkbench",
        page_icon="🔬",
        layout="wide",
    )

    st.sidebar.title("VisionWorkbench")
    st.sidebar.caption("v0.2 — 数据集中心")

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

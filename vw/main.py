"""VisionWorkbench Streamlit 入口."""

import streamlit as st

from vw.app import home


def main():
    st.set_page_config(
        page_title="VisionWorkbench",
        page_icon="🔬",
        layout="wide",
    )
    home.render()


if __name__ == "__main__":
    main()

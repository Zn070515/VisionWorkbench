"""概览页 — 单模型图片推理."""

import tempfile
from pathlib import Path

import cv2
import streamlit as st

from vw.core.model.inference import InferenceEngine
from vw.utils.image import load_image, bgr_to_pil, resize_to_max


def init_session():
    if "engine" not in st.session_state:
        st.session_state.engine = InferenceEngine()
    if "detections" not in st.session_state:
        st.session_state.detections = []
    if "annotated_image" not in st.session_state:
        st.session_state.annotated_image = None


def render_controls() -> dict:
    """渲染模型加载和参数控件，返回推理参数."""
    with st.expander("模型 & 参数", expanded=True):
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            model_file = st.file_uploader(
                "加载 YOLO 模型 (.pt)",
                type=["pt"],
                key="model_uploader",
            )

            if model_file is not None:
                with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                    tmp.write(model_file.read())
                    tmp_path = tmp.name

                engine: InferenceEngine = st.session_state.engine
                if engine.model_path != tmp_path:
                    with st.spinner("加载模型中..."):
                        info = engine.load(tmp_path)
                    st.success(f"已加载: {Path(tmp_path).name} ({info.get('num_classes', '?')} 类)")

            elif st.session_state.engine.is_loaded:
                engine = st.session_state.engine
                st.info(f"当前模型: {Path(engine.model_path).name}")
            else:
                st.warning("请上传 .pt 模型文件")
                return {}

        with col2:
            confidence = st.slider("置信度", 0.0, 1.0, 0.5, 0.05, key="conf_slider")
        with col3:
            device = st.selectbox("设备", ["cpu", "cuda:0"], key="device_select")

    return {"confidence": confidence, "device": device}


def render():
    init_session()

    st.title("概览")
    st.caption("单模型图片推理")

    params = render_controls()

    if not st.session_state.engine.is_loaded:
        return

    engine: InferenceEngine = st.session_state.engine

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("输入")
        source_type = st.radio("来源", ["图片文件", "摄像头"], horizontal=True)

        image = None
        if source_type == "图片文件":
            uploaded = st.file_uploader("选择图片", type=["jpg", "jpeg", "png", "bmp"])
            if uploaded:
                with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as tmp:
                    tmp.write(uploaded.read())
                    image = load_image(tmp.name)
        else:
            cam = st.camera_input("拍摄")
            if cam:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp.write(cam.read())
                    image = load_image(tmp.name)

        if image is not None:
            st.caption(f"尺寸: {image.shape[1]}×{image.shape[0]}")
            if st.button("运行推理", type="primary", use_container_width=True):
                with st.spinner("推理中..."):
                    display_img = resize_to_max(image, 1024)
                    detections = engine.predict(
                        display_img,
                        confidence=params.get("confidence", 0.5),
                        device=params.get("device", "cpu"),
                    )
                    st.session_state.detections = detections
                    st.session_state.annotated_image = engine.draw_detections(display_img, detections)
                st.rerun()

    with col2:
        st.subheader("结果")
        if st.session_state.annotated_image is not None:
            st.image(bgr_to_pil(st.session_state.annotated_image), use_container_width=True)
            detections = st.session_state.detections
            st.caption(f"检测到 {len(detections)} 个目标")

            if detections:
                with st.expander(f"检测列表 ({len(detections)})"):
                    for i, d in enumerate(detections):
                        st.text(f"#{i+1} {d['class_name']} ({d['confidence']:.2%})")

            if st.button("保存结果"):
                out_path = Path.home() / "visionworkbench" / "results"
                out_path.mkdir(parents=True, exist_ok=True)
                out_file = out_path / "inference_result.jpg"
                cv2.imwrite(str(out_file), st.session_state.annotated_image)
                st.success(f"已保存到 {out_file}")
        else:
            st.info("运行推理后结果将显示在这里")

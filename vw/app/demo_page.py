"""v1.0 演示页面 — 图片/视频/摄像头一键演示."""

import tempfile
import time
from pathlib import Path

import cv2
import streamlit as st

from vw.core.model.registry import ModelRegistry
from vw.core.model.inference import InferenceEngine
from vw.utils.image import load_image, bgr_to_pil, resize_to_max


def init_session():
    if "demo_engine" not in st.session_state:
        st.session_state.demo_engine = InferenceEngine()
    if "demo_model_reg" not in st.session_state:
        st.session_state.demo_model_reg = ModelRegistry()
    if "demo_result" not in st.session_state:
        st.session_state.demo_result = None
    if "demo_detections" not in st.session_state:
        st.session_state.demo_detections = []


def render_sidebar():
    st.sidebar.markdown("---")
    st.sidebar.subheader("演示配置")

    mode = st.sidebar.radio("输入模式", ["图片", "视频", "摄像头"], key="demo_mode")
    st.session_state.demo_mode = mode

    # 模型选择
    model_reg: ModelRegistry = st.session_state.demo_model_reg
    models = model_reg.list_all()
    if models:
        model_opts = {}
        for m in models:
            versions = model_reg.get_versions(m["id"])
            for v in versions:
                default_mark = " ★" if m["is_default"] else ""
                label = f"{m['name']} v{v['version']}{default_mark}"
                model_opts[label] = v["file_path"]

        # 默认模型优先
        default_model = model_reg.get_default()
        default_index = 0
        if default_model:
            # 查找默认模型在列表中的位置
            for i, (label, _) in enumerate(model_opts.items()):
                if default_model["name"] in label:
                    default_index = i
                    break

        model_choice = st.sidebar.selectbox("模型", list(model_opts.keys()),
                                            index=default_index, key="demo_model")
        st.session_state.demo_model_path = model_opts[model_choice]
    else:
        st.sidebar.warning("请先注册模型")
        st.session_state.demo_model_path = None

    confidence = st.sidebar.slider("置信度", 0.0, 1.0, 0.5, 0.05, key="demo_conf")
    st.session_state.demo_confidence = confidence


def render_main():
    st.title("演示中心")
    st.caption("快速演示模型效果")

    mode = st.session_state.get("demo_mode", "图片")
    model_path = st.session_state.get("demo_model_path")
    conf = st.session_state.get("demo_confidence", 0.5)

    if not model_path:
        st.warning("请先在模型页面注册模型，然后在此选择")
        return

    engine: InferenceEngine = st.session_state.demo_engine
    # 模型仅在更换时重新加载
    if not engine.is_loaded or engine.model_path != model_path:
        with st.spinner("加载模型..."):
            try:
                engine.load(model_path)
            except ValueError as e:
                st.error(str(e))
                return
        st.success(f"已加载: {Path(model_path).name}")

    if mode == "图片":
        _render_image_demo(engine, conf)
    elif mode == "视频":
        _render_video_demo(engine, conf)
    else:
        _render_webcam_demo(engine, conf)


def _render_image_demo(engine: InferenceEngine, conf: float):
    uploaded = st.file_uploader("选择图片", type=["jpg", "jpeg", "png", "bmp"], key="demo_img")

    if uploaded:
        with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        image = load_image(tmp_path)
        if image is None:
            st.error("无法读取图片")
            return

        col1, col2 = st.columns(2)
        with col1:
            st.image(bgr_to_pil(image), caption="原始图片", use_container_width=True)
        with col2:
            if st.button("运行推理", type="primary", use_container_width=True):
                display = resize_to_max(image, 1024)
                detections = engine.predict(display, confidence=conf)
                annotated = engine.draw_detections(display, detections)
                st.session_state.demo_result = annotated
                st.session_state.demo_detections = detections
                st.rerun()

            if st.session_state.demo_result is not None:
                st.image(bgr_to_pil(st.session_state.demo_result),
                         caption=f"检测到 {len(st.session_state.demo_detections)} 个目标",
                         use_container_width=True)

        if st.session_state.demo_detections:
            st.markdown("---")
            st.write(f"**检测结果 ({len(st.session_state.demo_detections)}):**")
            for i, d in enumerate(st.session_state.demo_detections):
                st.text(f"#{i+1} {d['class_name']} — {d['confidence']:.2%}")


def _render_video_demo(engine: InferenceEngine, conf: float):
    video_file = st.file_uploader("选择视频", type=["mp4", "avi", "mov", "mkv"], key="demo_video")

    if video_file:
        with tempfile.NamedTemporaryFile(suffix=Path(video_file.name).suffix, delete=False) as tmp:
            tmp.write(video_file.read())
            video_path = tmp.name

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.release()

        st.caption(f"总帧数: {total_frames} | FPS: {fps:.1f}")

        if st.button("逐帧分析", type="primary", use_container_width=True):
            cap = cv2.VideoCapture(video_path)
            frames_data = []
            progress = st.progress(0)
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % max(1, int(fps)) == 0:  # 每秒采样
                    display = resize_to_max(frame, 640)
                    detections = engine.predict(display, confidence=conf)
                    frames_data.append({
                        "frame": frame_idx,
                        "time": f"{frame_idx / fps:.1f}s",
                        "count": len(detections),
                        "image": engine.draw_detections(display, detections),
                    })
                frame_idx += 1
                progress.progress(min(frame_idx / total_frames, 1.0))

            cap.release()
            st.session_state.demo_video_frames = frames_data
            st.rerun()

        if st.session_state.get("demo_video_frames"):
            frames_data = st.session_state.demo_video_frames
            st.write(f"**采样 {len(frames_data)} 帧**")

            frame_idx = st.slider("选择帧", 0, len(frames_data) - 1, 0, key="demo_vframe_slider")
            fdata = frames_data[frame_idx]
            st.image(
                bgr_to_pil(fdata["image"]),
                caption=f"帧 {fdata['frame']} | {fdata['time']} | {fdata['count']} 个目标",
                use_container_width=True,
            )


def _render_webcam_demo(engine: InferenceEngine, conf: float):
    st.info("摄像头模式需要浏览器权限。点击下方按钮拍照。")

    cam = st.camera_input("拍摄照片", key="demo_cam")

    if cam:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(cam.read())
            tmp_path = tmp.name

        image = load_image(tmp_path)
        if image is not None:
            display = resize_to_max(image, 1024)
            detections = engine.predict(display, confidence=conf)
            annotated = engine.draw_detections(display, detections)

            st.image(bgr_to_pil(annotated), caption=f"检测到 {len(detections)} 个目标",
                     use_container_width=True)

            if detections:
                with st.expander(f"检测详情 ({len(detections)})"):
                    for i, d in enumerate(detections):
                        st.text(f"#{i+1} {d['class_name']} ({d['confidence']:.2%})")


def render():
    init_session()
    render_sidebar()
    render_main()

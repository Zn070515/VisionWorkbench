"""v0.5 实验页面 — 创建、对比、记录结论."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from vw.core.experiment.manager import ExperimentManager


def init_session():
    if "em" not in st.session_state:
        st.session_state.em = ExperimentManager()
    if "exp_compare_ids" not in st.session_state:
        st.session_state.exp_compare_ids = set()


def render_sidebar():
    st.sidebar.markdown("---")
    st.sidebar.subheader("新建实验")

    name = st.sidebar.text_input("实验名称", key="exp_name")
    desc = st.sidebar.text_area("描述", key="exp_desc", height=68)
    hypothesis = st.sidebar.text_area("假设", key="exp_hyp", height=68,
                                      placeholder="预期结果是什么？")
    tag_str = st.sidebar.text_input("标签 (逗号分隔)", key="exp_tags",
                                    placeholder="augmentation, lr_sweep")

    if st.sidebar.button("创建实验", use_container_width=True, disabled=not name):
        tags = [t.strip() for t in tag_str.split(",") if t.strip()] if tag_str else []
        em: ExperimentManager = st.session_state.em
        eid = em.create(name=name, description=desc or None,
                        hypothesis=hypothesis or None, tags=tags)
        st.session_state.selected_experiment_id = eid
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("实验列表")

    em: ExperimentManager = st.session_state.em
    experiments = em.list_all()

    if experiments:
        for exp in experiments:
            tags_display = f" [{', '.join(exp.get('tags', []))}]" if exp.get("tags") else ""
            label = f"{exp['name']}{tags_display}"
            if st.sidebar.button(label, key=f"exp_{exp['id']}", use_container_width=True):
                st.session_state.selected_experiment_id = exp["id"]
                st.rerun()
    else:
        st.sidebar.info("暂无实验")

    st.sidebar.markdown("---")
    st.sidebar.caption("对比模式下，选择多个实验进行对比")


def render_main():
    st.title("实验中心")
    st.caption("创建实验、关联训练任务、对比指标、记录结论")

    if "selected_experiment_id" not in st.session_state:
        st.info("👈 创建实验或选择已有实验")
        return

    em: ExperimentManager = st.session_state.em
    exp = em.get(st.session_state.selected_experiment_id)
    if not exp:
        st.error("实验不存在")
        return

    # ── 基本信息 ──
    col1, col2, col3 = st.columns(3)
    col1.metric("名称", exp["name"])
    tasks = em.get_tasks(exp["id"])
    col2.metric("关联任务", str(len(tasks)))
    col3.metric("标签", ", ".join(exp.get("tags", [])) or "-")

    if exp.get("description"):
        st.caption(f"描述: {exp['description']}")
    if exp.get("hypothesis"):
        st.caption(f"假设: {exp['hypothesis']}")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["指标对比", "关联任务", "笔记 & 结论"])

    with tab1:
        _render_comparison(exp)

    with tab2:
        _render_task_linking(exp)

    with tab3:
        _render_notes(exp)


def _render_comparison(exp: dict):
    """多选实验指标对比."""
    em: ExperimentManager = st.session_state.em
    all_exps = em.list_all()

    exp_options = {f"{e['name']} (ID:{e['id']})": e["id"] for e in all_exps}
    selected = st.multiselect(
        "选择实验进行对比",
        list(exp_options.keys()),
        default=[k for k, v in exp_options.items() if v == exp["id"]],
        key="exp_compare_select",
    )

    if len(selected) < 2:
        st.info("选择 2 个或以上实验查看对比图表")
        # 即使单选也展示该实验任务的指标
        if len(selected) == 1:
            eid = exp_options[selected[0]]
            comp = em.get_comparison([eid])
            _render_single_experiment_metrics(comp["experiments"])
        return

    exp_ids = [exp_options[s] for s in selected]
    comp = em.get_comparison(exp_ids)

    st.subheader("最佳指标对比")

    exps_data = comp["experiments"]
    names = [e["name"] for e in exps_data]

    # 柱状图
    fig = go.Figure()
    for metric, mname in [("best_map50", "mAP50"), ("best_map50_95", "mAP50-95"),
                           ("best_precision", "Precision"), ("best_recall", "Recall")]:
        vals = [e[metric] for e in exps_data]
        if any(v > 0 for v in vals):
            fig.add_trace(go.Bar(name=mname, x=names, y=vals))
    fig.update_layout(height=400, barmode="group",
                      xaxis_title="实验", yaxis_title="值")
    st.plotly_chart(fig, use_container_width=True)

    # 汇总表
    rows = []
    for e in exps_data:
        rows.append({
            "实验": e["name"],
            "任务数": e["task_count"],
            "mAP50": f"{e['best_map50']:.4f}",
            "mAP50-95": f"{e['best_map50_95']:.4f}",
            "假设": (e.get("hypothesis") or "")[:60],
            "结论": (e.get("conclusion") or "")[:60],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # 按任务展开的 epoch 级对比
    st.subheader("Epoch 级指标对比")
    all_tasks = []
    for e in exps_data:
        for t in e["tasks"]:
            if t["metrics"]:
                all_tasks.append({
                    "experiment": e["name"],
                    "task": t["model_name"],
                    "metrics": t["metrics"],
                })

    if all_tasks:
        metric_choice = st.selectbox("选择指标", ["map50", "map50_95", "box_loss", "cls_loss"])
        fig = go.Figure()
        for t in all_tasks:
            epochs = [m["epoch"] for m in t["metrics"]]
            vals = [m.get(metric_choice) for m in t["metrics"]]
            label = f"{t['experiment']}/{t['task']}"
            fig.add_trace(go.Scatter(x=epochs, y=vals, mode="lines+markers", name=label))
        fig.update_layout(height=400, xaxis_title="Epoch", yaxis_title=metric_choice)
        st.plotly_chart(fig, use_container_width=True)


def _render_single_experiment_metrics(experiments: list[dict]):
    """单实验指标展示."""
    if not experiments:
        return
    e = experiments[0]
    if not e["tasks"]:
        st.info("该实验未关联训练任务")
        return

    st.subheader(f"{e['name']} — 训练指标")
    fig = go.Figure()
    for t in e["tasks"]:
        if t["metrics"]:
            epochs = [m["epoch"] for m in t["metrics"]]
            map50_vals = [m.get("map50", 0) or 0 for m in t["metrics"]]
            fig.add_trace(go.Scatter(
                x=epochs, y=map50_vals, mode="lines+markers",
                name=f"{t['model_name']} (mAP50)",
            ))
    if fig.data:
        fig.update_layout(height=400, xaxis_title="Epoch", yaxis_title="mAP50")
        st.plotly_chart(fig, use_container_width=True)

    # 任务摘要表
    rows = []
    for t in e["tasks"]:
        rows.append({
            "任务": t["model_name"],
            "基础模型": t["base_model"],
            "状态": t["status"],
            "最优 mAP50": f"{t['best_map50']:.4f}",
            "完成 Epochs": t["epochs_completed"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_task_linking(exp: dict):
    """关联/取消关联训练任务."""
    em: ExperimentManager = st.session_state.em

    # 已关联任务
    linked = em.get_tasks(exp["id"])
    if linked:
        st.write("**已关联任务:**")
        for t in linked:
            col1, col2, col3 = st.columns([3, 1, 1])
            col1.text(f"{t['model_name']} ({t['status']}) — {t.get('created_at', '')[:16]}")
            col2.text(f"ID:{t['id']}")
            if col3.button("取消关联", key=f"unlink_{t['id']}"):
                em.unlink_task(exp["id"], t["id"])
                st.rerun()
    else:
        st.info("尚未关联训练任务")

    st.markdown("---")
    st.write("**关联新任务:**")
    available = em.get_available_tasks()
    unlinked = [t for t in available if t["id"] not in {l["id"] for l in linked}]

    if unlinked:
        task_options = {
            f"{t['model_name']} ({t['status']}) — ID:{t['id']}": t["id"]
            for t in unlinked
        }
        selected = st.selectbox("选择训练任务", list(task_options.keys()), key="link_task_select")
        if st.button("关联任务"):
            em.link_task(exp["id"], task_options[selected])
            st.rerun()
    else:
        st.caption("没有可关联的新任务")


def _render_notes(exp: dict):
    """编辑实验笔记和结论."""
    em: ExperimentManager = st.session_state.em

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("假设")
        new_hypothesis = st.text_area(
            "Hypothesis", value=exp.get("hypothesis") or "",
            height=120, key="edit_hypothesis", label_visibility="collapsed",
            placeholder="本次实验的假设是什么？",
        )
    with col2:
        st.subheader("结论")
        new_conclusion = st.text_area(
            "Conclusion", value=exp.get("conclusion") or "",
            height=120, key="edit_conclusion", label_visibility="collapsed",
            placeholder="实验结论和后续计划...",
        )

    col_a, col_b = st.columns([1, 3])
    with col_a:
        if st.button("保存笔记", type="primary", use_container_width=True):
            em.update(
                exp["id"],
                hypothesis=new_hypothesis or None,
                conclusion=new_conclusion or None,
            )
            st.success("已保存")
            st.rerun()

    st.markdown("---")
    if st.button("删除实验", type="secondary"):
        em.delete(exp["id"])
        del st.session_state.selected_experiment_id
        st.rerun()


def render():
    init_session()
    render_sidebar()
    render_main()

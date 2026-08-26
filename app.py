import json
import streamlit as st
import plotly.graph_objects as go
from models.schema import Qubit, Connection, ChipConstraints, Project
from core.scoring import compute_lqs, explain_risk
from core.q_drc import validate_layout
from core.physics_engine import build_risk_results
from core.optimizer import optimize_layout

st.set_page_config(page_title="Q-Layout", layout="wide")

CHIP_WIDTH = 100.0
CHIP_HEIGHT = 100.0

COLORS = {
    "bg": "#0f172a",
    "high_risk": "#ef4444",
    "medium_risk": "#f59e0b",
    "low_risk": "#10b981",
    "text": "#e2e8f0",
    "grid": "#334155",
    "intended": "#60a5fa",
}


def _make_constraints(**overrides):
    defaults = {
        "min_qubit_spacing_um": 10.0,
        "min_frequency_separation_mhz": 50.0,
        "frequency_check_distance_um": 20.0,
        "min_boundary_clearance_um": 5.0,
    }
    defaults.update(overrides)
    return ChipConstraints(**defaults)


def _make_demo_dataset():
    qubits = [
        Qubit(id="Q0", x_um=10.0, y_um=10.0, frequency_mhz=5000.0, movable=True),
        Qubit(id="Q1", x_um=20.0, y_um=10.0, frequency_mhz=5100.0, movable=False),
        Qubit(id="Q2", x_um=50.0, y_um=50.0, frequency_mhz=5200.0, movable=True),
        Qubit(id="Q3", x_um=85.0, y_um=50.0, frequency_mhz=5300.0, movable=True),
        Qubit(id="Q4", x_um=35.0, y_um=10.0, frequency_mhz=5105.0, movable=True),
        Qubit(id="Q5", x_um=60.5, y_um=50.0, frequency_mhz=5255.0, movable=True),
        Qubit(id="Q6", x_um=80.0, y_um=80.0, frequency_mhz=5400.0, movable=True),
        Qubit(id="Q7", x_um=90.0, y_um=80.0, frequency_mhz=5500.0, movable=True),
    ]
    connections = [
        Connection(source_qubit_id="Q0", target_qubit_id="Q1", interaction_weight=1.0, gate_count=10),
        Connection(source_qubit_id="Q0", target_qubit_id="Q2", interaction_weight=0.0, gate_count=5),
        Connection(source_qubit_id="Q2", target_qubit_id="Q5", interaction_weight=0.0, gate_count=5),
        Connection(source_qubit_id="Q1", target_qubit_id="Q4", interaction_weight=0.0, gate_count=5),
    ]
    return qubits, connections


def _get_project():
    return st.session_state.get("project")


def _set_project(project):
    st.session_state["project"] = project


def _init_session_state():
    if "project" not in st.session_state:
        qubits, connections = _make_demo_dataset()
        constraints = _make_constraints()
        project = Project(
            id="demo",
            name="Demo Chip",
            chip_width_um=CHIP_WIDTH,
            chip_height_um=CHIP_HEIGHT,
            constraints=constraints,
            qubits=qubits,
            connections=connections,
        )
        _set_project(project)


def _apply_movements(qubits, movements):
    positions = {q.id: (q.x_um, q.y_um) for q in qubits}
    for move in movements:
        positions[move["qubit_id"]] = (move["to_x"], move["to_y"])
    return [
        Qubit(q.id, positions[q.id][0], positions[q.id][1], q.frequency_mhz, q.movable)
        for q in qubits
    ]


def _build_canvas(project):
    qubits = project.qubits
    connections = project.connections
    chip_width = project.chip_width_um
    chip_height = project.chip_height_um

    fig = go.Figure()

    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=chip_width, y1=chip_height,
        line=dict(color=COLORS["grid"], width=2),
        fillcolor="rgba(15, 23, 42, 0.5)",
        layer="below",
    )

    risk_results = build_risk_results(qubits, connections)
    for rr in risk_results:
        q_a = next(q for q in qubits if q.id == rr.source_qubit_id)
        q_b = next(q for q in qubits if q.id == rr.target_qubit_id)

        if rr.interaction_weight > 0.0:
            color = COLORS["intended"]
            dash = "solid"
            width = 1.5
        else:
            if rr.objective_penalty > 0.65:
                color = COLORS["high_risk"]
            elif rr.objective_penalty >= 0.3:
                color = COLORS["medium_risk"]
            else:
                color = COLORS["low_risk"]
            dash = "dash"
            width = 2.0

        fig.add_trace(go.Scatter(
            x=[q_a.x_um, q_b.x_um],
            y=[q_a.y_um, q_b.y_um],
            mode="lines",
            line=dict(color=color, width=width, dash=dash),
            hoverinfo="none",
            showlegend=False,
        ))

    q_x = [q.x_um for q in qubits]
    q_y = [q.y_um for q in qubits]
    q_text = [
        f"{q.id}<br>x={q.x_um:.1f}, y={q.y_um:.1f}<br>f={q.frequency_mhz:.0f} MHz"
        for q in qubits
    ]

    fig.add_trace(go.Scatter(
        x=q_x,
        y=q_y,
        mode="markers+text",
        marker=dict(size=16, color=COLORS["text"], line=dict(width=2, color=COLORS["bg"])),
        text=[q.id for q in qubits],
        textposition="top center",
        hovertext=q_text,
        hoverinfo="text",
        showlegend=False,
    ))

    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"]),
        xaxis=dict(
            range=[0, chip_width],
            showgrid=True,
            gridcolor=COLORS["grid"],
            zeroline=False,
            showticklabels=True,
            title="X (um)",
        ),
        yaxis=dict(
            range=[0, chip_height],
            showgrid=True,
            gridcolor=COLORS["grid"],
            zeroline=False,
            showticklabels=True,
            title="Y (um)",
            scaleanchor="x",
            scaleratio=1,
        ),
        margin=dict(l=40, r=40, t=40, b=40),
        height=500,
    )

    return fig


def _build_lqs_panel(project):
    qubits = project.qubits
    connections = project.connections

    st.subheader("LQS Score")
    lqs = compute_lqs(qubits, connections)
    st.metric("LQS", f"{lqs:.1f}")


def _build_risk_panel(project):
    qubits = project.qubits
    connections = project.connections

    st.subheader("Risk Panel")

    risk_results = build_risk_results(qubits, connections)
    if not risk_results:
        st.write("No connections defined.")
        return

    for rr in risk_results:
        if rr.interaction_weight > 0.0:
            severity = "LOW"
            color = COLORS["low_risk"]
            icon = "✓"
        else:
            severity = rr.severity
            if severity == "HIGH":
                color = COLORS["high_risk"]
                icon = "🔴"
            elif severity == "MEDIUM":
                color = COLORS["medium_risk"]
                icon = "⚠"
            else:
                color = COLORS["low_risk"]
                icon = "✓"

        st.markdown(
            f"<span style='color:{color}'>{icon} {rr.source_qubit_id}-{rr.target_qubit_id}</span> "
            f"(penalty={rr.objective_penalty:.2f})",
            unsafe_allow_html=True,
        )


def _build_drc_panel(project):
    qubits = project.qubits
    connections = project.connections
    constraints = project.constraints
    chip_width = project.chip_width_um
    chip_height = project.chip_height_um

    st.subheader("Q-DRC Warnings")
    drc = validate_layout(qubits, constraints, connections, chip_width, chip_height)

    if not drc.violations and not drc.warnings:
        st.success("No violations or warnings.")
    else:
        for v in drc.violations:
            st.error(f"**Violation:** {v.get('rule', 'unknown')} - {v.get('detail', '')}")
        for w in drc.warnings:
            st.warning(f"**Warning:** {w.get('rule', 'unknown')} - {w.get('detail', '')}")


def _build_explainability_panel(project):
    qubits = project.qubits
    connections = project.connections

    st.subheader("Risk Explainability")

    risk_results = build_risk_results(qubits, connections)
    if not risk_results:
        st.write("No connections defined.")
        return

    options = [f"{rr.source_qubit_id}-{rr.target_qubit_id}" for rr in risk_results]
    selected_pair = st.selectbox("Select a pair", options=options)

    for rr in risk_results:
        if f"{rr.source_qubit_id}-{rr.target_qubit_id}" == selected_pair:
            explanation = explain_risk(rr)
            st.write(f"**Severity:** {explanation['severity']}")
            st.write(f"**Penalty:** {explanation['objective_penalty']:.4f}")
            for reason in explanation["reasons"]:
                st.write(f"- {reason}")
            break


def main():
    _init_session_state()
    project = _get_project()

    st.title("Q-Layout")
    st.caption(f"Project: {project.name}")

    with st.sidebar:
        st.header("Project")
        if st.button("Load Demo Chip"):
            qubits, connections = _make_demo_dataset()
            constraints = _make_constraints()
            project = Project(
                id="demo",
                name="Demo Chip",
                chip_width_um=CHIP_WIDTH,
                chip_height_um=CHIP_HEIGHT,
                constraints=constraints,
                qubits=qubits,
                connections=connections,
            )
            _set_project(project)
            st.rerun()

        st.subheader("Export")
        if st.button("Export Layout"):
            layout_data = {
                "id": project.id,
                "name": project.name,
                "chip_width_um": project.chip_width_um,
                "chip_height_um": project.chip_height_um,
                "constraints": {
                    "min_qubit_spacing_um": project.constraints.min_qubit_spacing_um,
                    "min_frequency_separation_mhz": project.constraints.min_frequency_separation_mhz,
                    "frequency_check_distance_um": project.constraints.frequency_check_distance_um,
                    "min_boundary_clearance_um": project.constraints.min_boundary_clearance_um,
                },
                "qubits": [
                    {
                        "id": q.id,
                        "x_um": q.x_um,
                        "y_um": q.y_um,
                        "frequency_mhz": q.frequency_mhz,
                        "movable": q.movable,
                    }
                    for q in project.qubits
                ],
                "connections": [
                    {
                        "source_qubit_id": c.source_qubit_id,
                        "target_qubit_id": c.target_qubit_id,
                        "interaction_weight": c.interaction_weight,
                        "gate_count": c.gate_count,
                    }
                    for c in project.connections
                ],
            }
            st.download_button(
                label="Download JSON",
                data=json.dumps(layout_data, indent=2),
                file_name="layout.json",
                mime="application/json",
            )

    col_canvas, col_info = st.columns([3, 1])

    with col_canvas:
        fig = _build_canvas(project)
        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        _build_lqs_panel(project)
        _build_drc_panel(project)
        _build_explainability_panel(project)

    st.divider()
    col_lqs, col_drc, col_opt = st.columns([1, 1, 1])

    with col_lqs:
        lqs = compute_lqs(project.qubits, project.connections)
        st.metric("LQS", f"{lqs:.1f}")

    with col_drc:
        drc = validate_layout(
            project.qubits,
            project.constraints,
            project.connections,
            project.chip_width_um,
            project.chip_height_um,
        )
        drc_count = len(drc.violations) + len(drc.warnings)
        st.metric("DRC Issues", drc_count)

    with col_opt:
        if st.button("Auto-Optimize", type="primary"):
            result = optimize_layout(
                project.qubits,
                project.connections,
                project.constraints,
                project.chip_width_um,
                project.chip_height_um,
            )

            new_qubits = _apply_movements(project.qubits, result.movements)
            project.qubits = new_qubits
            _set_project(project)

            st.success(
                f"Optimization complete!\n"
                f"Before LQS: {result.before_lqs:.1f} → After LQS: {result.after_lqs:.1f}\n"
                f"Iterations: {result.iterations} | Stopped: {result.stopped_reason}"
            )


if __name__ == "__main__":
    main()

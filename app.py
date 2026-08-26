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


def _make_blank_project():
    return Project(
        id="blank",
        name="Blank Chip",
        chip_width_um=CHIP_WIDTH,
        chip_height_um=CHIP_HEIGHT,
        constraints=_make_constraints(),
        qubits=[],
        connections=[],
    )


def _get_project():
    return st.session_state.get("project")


def _set_project(project):
    st.session_state["project"] = project


def _next_qubit_id(project):
    used_ids = {q.id for q in project.qubits}
    index = 0
    while True:
        candidate = f"Q{index}"
        if candidate not in used_ids:
            return candidate
        index += 1


def _init_session_state():
    if "project" not in st.session_state:
        _set_project(_make_blank_project())


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

        if st.button("Create Blank Project"):
            _set_project(_make_blank_project())
            st.rerun()

        with st.expander("Configure Chip", expanded=True):
            project.name = st.text_input("Project name", value=project.name)
            project.chip_width_um = st.number_input(
                "Chip width (um)",
                min_value=10.0,
                value=float(project.chip_width_um),
                step=1.0,
            )
            project.chip_height_um = st.number_input(
                "Chip height (um)",
                min_value=10.0,
                value=float(project.chip_height_um),
                step=1.0,
            )
            project.constraints.min_qubit_spacing_um = st.number_input(
                "Min qubit spacing (um)",
                min_value=0.1,
                value=float(project.constraints.min_qubit_spacing_um),
                step=0.5,
            )
            project.constraints.min_frequency_separation_mhz = st.number_input(
                "Min frequency separation (MHz)",
                min_value=1.0,
                value=float(project.constraints.min_frequency_separation_mhz),
                step=1.0,
            )
            project.constraints.frequency_check_distance_um = st.number_input(
                "Frequency check distance (um)",
                min_value=0.1,
                value=float(project.constraints.frequency_check_distance_um),
                step=0.5,
            )
            project.constraints.min_boundary_clearance_um = st.number_input(
                "Min boundary clearance (um)",
                min_value=0.0,
                value=float(project.constraints.min_boundary_clearance_um),
                step=0.5,
            )
            _set_project(project)

        with st.expander("Qubit Manager", expanded=True):
            if not project.qubits:
                st.info("No qubits yet. Add a few to begin layout analysis.")

            with st.form("add_qubit_form", clear_on_submit=True):
                qubit_id = st.text_input("Qubit ID", value=_next_qubit_id(project))
                x_um = st.number_input("X (um)", min_value=0.0, value=10.0, step=1.0)
                y_um = st.number_input("Y (um)", min_value=0.0, value=10.0, step=1.0)
                frequency_mhz = st.number_input("Frequency (MHz)", min_value=0.0, value=5000.0, step=10.0)
                movable = st.checkbox("Movable", value=True)
                submitted = st.form_submit_button("Add Qubit")

            if submitted:
                clean_id = qubit_id.strip() or _next_qubit_id(project)
                if any(existing.id == clean_id for existing in project.qubits):
                    st.warning(f"Qubit '{clean_id}' already exists.")
                else:
                    project.qubits.append(
                        Qubit(
                            id=clean_id,
                            x_um=float(x_um),
                            y_um=float(y_um),
                            frequency_mhz=float(frequency_mhz),
                            movable=movable,
                        )
                    )
                    _set_project(project)
                    st.success(f"Added {clean_id}.")

            if project.qubits:
                st.markdown("**Current Qubits**")
                for q in project.qubits:
                    cols = st.columns([3, 1])
                    cols[0].write(f"{q.id}: ({q.x_um:.1f}, {q.y_um:.1f}) @ {q.frequency_mhz:.0f} MHz")
                    if cols[1].button("Remove", key=f"remove_qubit_{q.id}"):
                        project.qubits = [existing for existing in project.qubits if existing.id != q.id]
                        project.connections = [
                            conn for conn in project.connections
                            if conn.source_qubit_id != q.id and conn.target_qubit_id != q.id
                        ]
                        _set_project(project)
                        st.rerun()

        with st.expander("Connection Builder", expanded=True):
            qubit_ids = [q.id for q in project.qubits]
            if len(qubit_ids) < 2:
                st.info("Add at least two qubits to create a connection.")
            else:
                with st.form("add_connection_form", clear_on_submit=True):
                    source_qubit_id = st.selectbox("Source Qubit", options=qubit_ids)
                    target_qubit_id = st.selectbox(
                        "Target Qubit",
                        options=[q_id for q_id in qubit_ids if q_id != source_qubit_id],
                    )
                    interaction_weight = st.slider("Interaction weight (I_ij)", 0.0, 1.0, 0.5, 0.01)
                    gate_count = st.number_input("Gate count", min_value=1, value=1, step=1)
                    submitted = st.form_submit_button("Add Connection")

                if submitted:
                    if source_qubit_id == target_qubit_id:
                        st.warning("A qubit cannot be connected to itself.")
                    else:
                        pair_exists = any(
                            (
                                (conn.source_qubit_id == source_qubit_id and conn.target_qubit_id == target_qubit_id)
                                or (conn.source_qubit_id == target_qubit_id and conn.target_qubit_id == source_qubit_id)
                            )
                            for conn in project.connections
                        )
                        if pair_exists:
                            st.warning("This connection already exists.")
                        else:
                            project.connections.append(
                                Connection(
                                    source_qubit_id=source_qubit_id,
                                    target_qubit_id=target_qubit_id,
                                    interaction_weight=float(interaction_weight),
                                    gate_count=int(gate_count),
                                )
                            )
                            _set_project(project)
                            st.success(f"Connected {source_qubit_id} to {target_qubit_id}.")

            if project.connections:
                st.markdown("**Current Connections**")
                for conn in project.connections:
                    label = f"{conn.source_qubit_id} → {conn.target_qubit_id} (I={conn.interaction_weight:.2f})"
                    cols = st.columns([4, 1])
                    cols[0].write(label)
                    if cols[1].button("Remove", key=f"remove_conn_{conn.source_qubit_id}_{conn.target_qubit_id}"):
                        project.connections = [
                            existing for existing in project.connections
                            if not (
                                existing.source_qubit_id == conn.source_qubit_id
                                and existing.target_qubit_id == conn.target_qubit_id
                            )
                        ]
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

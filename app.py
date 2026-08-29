import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from models.schema import Qubit, Connection, ChipConstraints, Project
from core.scoring import compute_lqs, explain_risk, get_severity_from_penalty
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

def _init_session_state():
    if "project" not in st.session_state:
        _set_project(_make_blank_project())
    if "editor_version" not in st.session_state:
        st.session_state["editor_version"] = 0
    if "candidate_project" not in st.session_state:
        st.session_state["candidate_project"] = None
    if "last_optimization_result" not in st.session_state:
        st.session_state["last_optimization_result"] = None

def _apply_movements(qubits, movements):
    positions = {q.id: (q.x_um, q.y_um) for q in qubits}
    for move in movements:
        positions[move["qubit_id"]] = (move["to_x"], move["to_y"])
    return [
        Qubit(q.id, positions[q.id][0], positions[q.id][1], q.frequency_mhz, q.movable)
        for q in qubits
    ]

def _build_canvas(project, show_intended=True, show_high=True, show_medium=True, show_low=False):
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
        q_a = next((q for q in qubits if q.id == rr.source_qubit_id), None)
        q_b = next((q for q in qubits if q.id == rr.target_qubit_id), None)
        
        if not q_a or not q_b:
            continue

        is_intended = rr.interaction_weight > 0.0
        
        draw = False
        if is_intended and show_intended:
            draw = True
        if rr.severity == "HIGH" and show_high:
            draw = True
        if rr.severity == "MEDIUM" and show_medium:
            draw = True
        if rr.severity == "LOW" and show_low:
            draw = True
            
        if not draw:
            continue

        if is_intended:
            color = COLORS["intended"]
            dash = "solid"
            width = 1.5
            if rr.severity in ["HIGH", "MEDIUM"] and ((rr.severity == "HIGH" and show_high) or (rr.severity == "MEDIUM" and show_medium)):
                bg_color = COLORS["high_risk"] if rr.severity == "HIGH" else COLORS["medium_risk"]
                fig.add_trace(go.Scatter(
                    x=[q_a.x_um, q_b.x_um],
                    y=[q_a.y_um, q_b.y_um],
                    mode="lines",
                    line=dict(color=bg_color, width=4.0, dash="dash"),
                    hoverinfo="none",
                    showlegend=False,
                ))
        else:
            if rr.severity == "HIGH":
                color = COLORS["high_risk"]
            elif rr.severity == "MEDIUM":
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

    if not qubits:
        fig.add_annotation(
            text="No qubits defined.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color=COLORS["text"])
        )
    else:
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
        height=600,
    )

    return fig

def render_design_tab(project):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load Demo Chip", use_container_width=True):
            qubits, connections = _make_demo_dataset()
            constraints = _make_constraints()
            p = Project(
                id="demo",
                name="Demo Chip",
                chip_width_um=CHIP_WIDTH,
                chip_height_um=CHIP_HEIGHT,
                constraints=constraints,
                qubits=qubits,
                connections=connections,
            )
            _set_project(p)
            st.session_state["editor_version"] += 1
            st.rerun()

    with col2:
        if st.button("Create Blank Project", use_container_width=True):
            _set_project(_make_blank_project())
            st.session_state["editor_version"] += 1
            st.rerun()

    st.divider()
    
    st.markdown("### Qubits")
    df_qubits = pd.DataFrame([{
        "id": q.id, "x_um": q.x_um, "y_um": q.y_um, "frequency_mhz": q.frequency_mhz, "movable": q.movable
    } for q in project.qubits])
    if df_qubits.empty:
        df_qubits = pd.DataFrame(columns=["id", "x_um", "y_um", "frequency_mhz", "movable"])

    edited_df_qubits = st.data_editor(
        df_qubits,
        num_rows="dynamic",
        key=f"qubit_editor_{st.session_state.editor_version}",
        use_container_width=True,
        column_config={
            "id": st.column_config.TextColumn("Qubit ID", required=True),
            "x_um": st.column_config.NumberColumn("X (um)", required=True),
            "y_um": st.column_config.NumberColumn("Y (um)", required=True),
            "frequency_mhz": st.column_config.NumberColumn("Freq (MHz)", required=True),
            "movable": st.column_config.CheckboxColumn("Movable", default=True),
        }
    )

    st.markdown("### Connections")
    df_conns = pd.DataFrame([{
        "source": c.source_qubit_id, "target": c.target_qubit_id, 
        "interaction_weight": c.interaction_weight, "gate_count": c.gate_count
    } for c in project.connections])
    if df_conns.empty:
        df_conns = pd.DataFrame(columns=["source", "target", "interaction_weight", "gate_count"])

    edited_df_conns = st.data_editor(
        df_conns,
        num_rows="dynamic",
        key=f"conn_editor_{st.session_state.editor_version}",
        use_container_width=True,
        column_config={
            "source": st.column_config.TextColumn("Source", required=True),
            "target": st.column_config.TextColumn("Target", required=True),
            "interaction_weight": st.column_config.NumberColumn("Interaction Weight", min_value=0.0, max_value=1.0, required=True),
            "gate_count": st.column_config.NumberColumn("Gate Count", min_value=1, step=1, required=True),
        }
    )

    new_qubits = []
    qubit_errors = []
    seen_q_ids = set()
    
    for i, row in edited_df_qubits.iterrows():
        qid = str(row.get("id", "")).strip()
        if qid == "nan" or qid == "None": qid = ""
        
        try:
            x = float(row.get("x_um", 0.0))
            if pd.isna(x): x = 0.0
            y = float(row.get("y_um", 0.0))
            if pd.isna(y): y = 0.0
            f = float(row.get("frequency_mhz", 5000.0))
            if pd.isna(f): f = 5000.0
            mov = bool(row.get("movable", True))
        except (ValueError, TypeError):
            qubit_errors.append(f"Row {i}: Numeric conversion error.")
            continue
            
        if not qid:
            qubit_errors.append(f"Row {i}: Qubit ID cannot be empty.")
            continue
        if qid in seen_q_ids:
            qubit_errors.append(f"Row {i}: Duplicate Qubit ID '{qid}'.")
            continue
        seen_q_ids.add(qid)
        
        try:
            q = Qubit(id=qid, x_um=x, y_um=y, frequency_mhz=f, movable=mov)
            new_qubits.append(q)
        except ValueError as e:
            qubit_errors.append(f"Row {i} ({qid}): {e}")

    new_conns = []
    conn_errors = []
    seen_pairs = set()
    
    for i, row in edited_df_conns.iterrows():
        src = str(row.get("source", "")).strip()
        if src == "nan" or src == "None": src = ""
        tgt = str(row.get("target", "")).strip()
        if tgt == "nan" or tgt == "None": tgt = ""
        
        try:
            w = float(row.get("interaction_weight", 0.5))
            if pd.isna(w): w = 0.5
            gc = int(row.get("gate_count", 1))
            if pd.isna(gc): gc = 1
        except (ValueError, TypeError):
            conn_errors.append(f"Row {i}: Numeric conversion error.")
            continue
            
        if not src or not tgt:
            conn_errors.append(f"Row {i}: Source and Target must not be empty.")
            continue
        if src not in seen_q_ids or tgt not in seen_q_ids:
            conn_errors.append(f"Row {i}: Connection references unknown qubit IDs ({src}, {tgt}).")
            continue
        if src == tgt:
            conn_errors.append(f"Row {i}: A qubit cannot connect to itself ({src}).")
            continue
            
        pair = frozenset([src, tgt])
        if pair in seen_pairs:
            conn_errors.append(f"Row {i}: Duplicate connection between {src} and {tgt}.")
            continue
        seen_pairs.add(pair)
        
        try:
            c = Connection(source_qubit_id=src, target_qubit_id=tgt, interaction_weight=w, gate_count=gc)
            new_conns.append(c)
        except ValueError as e:
            conn_errors.append(f"Row {i}: {e}")

    for e in qubit_errors:
        st.error(e)
    for e in conn_errors:
        st.error(e)
        
    if not qubit_errors and not conn_errors:
        project.qubits = new_qubits
        project.connections = new_conns

    st.divider()

    st.markdown("### Chip Constraints")
    c1, c2 = st.columns(2)
    new_name = st.text_input("Project name", value=project.name)
    new_w = c1.number_input("Chip width (um)", min_value=10.0, value=float(project.chip_width_um), step=1.0)
    new_h = c2.number_input("Chip height (um)", min_value=10.0, value=float(project.chip_height_um), step=1.0)
    
    cc1, cc2 = st.columns(2)
    min_q = cc1.number_input("Min qubit spacing (um)", value=float(project.constraints.min_qubit_spacing_um), step=0.5)
    min_f = cc2.number_input("Min frequency separation (MHz)", value=float(project.constraints.min_frequency_separation_mhz), step=1.0)
    f_dist = cc1.number_input("Frequency check distance (um)", value=float(project.constraints.frequency_check_distance_um), step=0.5)
    min_b = cc2.number_input("Min boundary clearance (um)", value=float(project.constraints.min_boundary_clearance_um), step=0.5)

    constraints_error = None
    try:
        new_c = ChipConstraints(
            min_qubit_spacing_um=min_q,
            min_frequency_separation_mhz=min_f,
            frequency_check_distance_um=f_dist,
            min_boundary_clearance_um=min_b
        )
    except ValueError as e:
        constraints_error = str(e)
        
    if constraints_error:
        st.error(f"Constraints Error: {constraints_error}")
    else:
        project.name = new_name
        project.chip_width_um = new_w
        project.chip_height_um = new_h
        project.constraints = new_c
        
    _set_project(project)

    st.divider()

    st.markdown("### 📍 Place Qubit on Canvas")
    q_ids = [q.id for q in project.qubits]
    if not q_ids:
        st.info("Create a qubit first to place it.")
    else:
        col_q, col_x, col_y, col_btn = st.columns([2, 1, 1, 1])
        place_q_id = col_q.selectbox("Qubit to place", options=q_ids)
        new_x = col_x.number_input("Target X", min_value=0.0, max_value=float(project.chip_width_um), value=10.0, step=1.0)
        new_y = col_y.number_input("Target Y", min_value=0.0, max_value=float(project.chip_height_um), value=10.0, step=1.0)
        
        if col_btn.button("Move Qubit", use_container_width=True):
            for q in project.qubits:
                if q.id == place_q_id:
                    q.x_um = new_x
                    q.y_um = new_y
            _set_project(project)
            st.session_state["editor_version"] += 1
            st.rerun()

    st.divider()

    st.markdown("### Selected Qubit")
    if not project.qubits:
        st.info("No qubits available.")
    else:
        selected_q_id = st.selectbox("Inspect Qubit", options=q_ids, key="inspect_q_id")
        if selected_q_id:
            q = next((q for q in project.qubits if q.id == selected_q_id), None)
            if q:
                st.write(f"**ID:** {q.id} | **Position:** ({q.x_um:.1f}, {q.y_um:.1f}) | **Frequency:** {q.frequency_mhz:.0f} MHz | **Movable:** {q.movable}")
                
                risk_results = build_risk_results(project.qubits, project.connections)
                q_risks = [r for r in risk_results if r.source_qubit_id == q.id or r.target_qubit_id == q.id]
                
                if q_risks:
                    nearest = min(q_risks, key=lambda r: r.distance_um)
                    worst = max(q_risks, key=lambda r: r.objective_penalty)
                    nearest_other = nearest.target_qubit_id if nearest.source_qubit_id == q.id else nearest.source_qubit_id
                    worst_other = worst.target_qubit_id if worst.source_qubit_id == q.id else worst.source_qubit_id
                    
                    st.write(f"- **Nearest qubit:** {nearest_other} ({nearest.distance_um:.2f} um)")
                    st.write(f"- **Worst interaction:** {worst_other} (Penalty: {worst.objective_penalty:.3f}, Severity: {worst.severity})")
                else:
                    st.write("No risk results involving this qubit (requires at least 2 connected/interacting qubits).")

    st.divider()
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

def render_analyze_tab(project):
    risk_results = build_risk_results(project.qubits, project.connections)
    drc = validate_layout(
        project.qubits,
        project.constraints,
        project.connections,
        project.chip_width_um,
        project.chip_height_um,
    )
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Qubits", len(project.qubits))
    col2.metric("Intended Connections", len(project.connections))
    
    high_risk_count = sum(1 for rr in risk_results if rr.severity == "HIGH")
    med_risk_count = sum(1 for rr in risk_results if rr.severity == "MEDIUM")
    
    col3.metric("High Risk Pairs", high_risk_count)
    col4.metric("Medium Risk Pairs", med_risk_count)
    
    if drc.violations:
        drc_status = "FAIL"
    elif drc.warnings:
        drc_status = "WARN"
    else:
        drc_status = "PASS"
    col5.metric("Q-DRC", drc_status)
    
    st.divider()

    st.markdown("### Risk Visibility Controls")
    c1, c2, c3, c4 = st.columns(4)
    show_intended = c1.checkbox("Intended", value=True)
    show_high = c2.checkbox("High Risk", value=True)
    show_medium = c3.checkbox("Medium Risk", value=True)
    show_low = c4.checkbox("Low Risk", value=False)
    
    col_canvas, col_info = st.columns([3, 1])

    with col_canvas:
        fig = _build_canvas(project, show_intended, show_high, show_medium, show_low)
        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        st.subheader("Metrics")
        lqs = compute_lqs(project.qubits, project.connections)
        st.metric("Layout Quality Score", f"{lqs:.1f} / 100")
        
        if not risk_results:
            st.metric("Worst Pair Risk", "N/A")
        else:
            worst_risk = max(risk_results, key=lambda r: r.objective_penalty)
            penalty = worst_risk.objective_penalty
            pair_str = f"{worst_risk.source_qubit_id} <-> {worst_risk.target_qubit_id}"
            
            if worst_risk.severity == "HIGH":
                st.metric("Worst Pair Risk", f"{penalty:.3f} (HIGH)")
                st.caption(f"**High Risk Pair:** {pair_str}")
            else:
                st.metric("Worst Pair Risk", f"{penalty:.3f}")
                st.caption(f"Highest-risk pair: {pair_str}")

        st.divider()

        st.subheader("Q-DRC Summary")
        if not drc.violations and not drc.warnings:
            st.success("PASS: No violations or warnings.")
        else:
            if drc.violations:
                st.error(f"FAIL: {len(drc.violations)} violations.")
            if drc.warnings:
                st.warning(f"WARN: {len(drc.warnings)} warnings.")

    st.divider()
    st.subheader("Risk Explanation")
    
    if not risk_results:
        st.info("Not enough qubits to analyze risk.")
    else:
        options = [f"{rr.source_qubit_id} <-> {rr.target_qubit_id}" for rr in risk_results]
        selected_pair = st.selectbox("Select a pair to inspect", options=options)

        for rr in risk_results:
            if f"{rr.source_qubit_id} <-> {rr.target_qubit_id}" == selected_pair:
                explanation = explain_risk(rr)
                
                c_a, c_b = st.columns(2)
                c_a.write(f"**Severity:** {explanation['severity']}")
                c_b.write(f"**Penalty:** {explanation['objective_penalty']:.4f}")
                
                st.markdown("**Reasons:**")
                for reason in explanation["reasons"]:
                    st.markdown(f"- {reason}")
                break

def render_optimize_tab(project):
    st.subheader("Optimizer Control")
    st.write("Run the deterministic 8-direction hill-climbing optimizer to improve the Layout Quality Score while respecting Q-DRC constraints.")

    if st.button("Auto-Optimize", type="primary"):
        with st.spinner("Optimizing layout..."):
            result = optimize_layout(
                project.qubits,
                project.connections,
                project.constraints,
                project.chip_width_um,
                project.chip_height_um,
            )

            new_qubits = _apply_movements(project.qubits, result.movements)
            
            candidate = Project(
                id=project.id,
                name=project.name,
                chip_width_um=project.chip_width_um,
                chip_height_um=project.chip_height_um,
                constraints=project.constraints,
                qubits=new_qubits,
                connections=project.connections,
            )
            st.session_state["candidate_project"] = candidate
            st.session_state["last_optimization_result"] = result
            st.rerun()

    st.divider()
    
    candidate = st.session_state.get("candidate_project")
    result = st.session_state.get("last_optimization_result")
    
    if candidate and result:
        st.subheader("Optimization Results")
        
        if result.movements:
            x_vals = [0] + [m["iteration"] for m in result.movements]
            y_vals = [result.lqs_before] + [m["lqs_after"] for m in result.movements]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines+markers', line=dict(color=COLORS['intended'])))
            fig.update_layout(
                title="Optimization Convergence",
                xaxis_title="Iteration",
                yaxis_title="LQS",
                paper_bgcolor=COLORS["bg"],
                plot_bgcolor=COLORS["bg"],
                font=dict(color=COLORS["text"]),
                margin=dict(l=20, r=20, t=40, b=20),
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            
        c1, c2 = st.columns(2)
        
        risk_before = build_risk_results(project.qubits, project.connections)
        worst_before = max((r.objective_penalty for r in risk_before), default=None)
        worst_before_str = f"{worst_before:.3f}" if worst_before is not None else "N/A"
        
        c1.markdown("#### BEFORE")
        c1.metric("LQS", f"{result.lqs_before:.1f}")
        c1.metric("Worst Risk", worst_before_str)
        
        risk_after = build_risk_results(candidate.qubits, candidate.connections)
        worst_after = max((r.objective_penalty for r in risk_after), default=None)
        worst_after_str = f"{worst_after:.3f}" if worst_after is not None else "N/A"
        
        c2.markdown("#### AFTER")
        c2.metric("LQS", f"{result.lqs_after:.1f}", delta=f"{result.lqs_after - result.lqs_before:.1f}")
        c2.metric("Worst Risk", worst_after_str)
        
        st.info(f"**Iterations:** {result.iterations} | **Stopped:** {result.stopped_reason}")
        
        c_apply, c_revert = st.columns(2)
        if c_apply.button("✅ Apply Candidate Layout", type="primary", use_container_width=True):
            _set_project(candidate)
            st.session_state["candidate_project"] = None
            st.session_state["last_optimization_result"] = None
            st.session_state["editor_version"] += 1
            st.rerun()
            
        if c_revert.button("❌ Revert", use_container_width=True):
            st.session_state["candidate_project"] = None
            st.session_state["last_optimization_result"] = None
            st.rerun()
    else:
        st.info("No optimization candidate yet.\n\nRun Auto-Optimize to generate a candidate layout.")

def main():
    _init_session_state()
    project = _get_project()

    st.title("Q-Layout")
    st.caption(f"Project: {project.name}")

    tab_design, tab_analyze, tab_optimize = st.tabs([
        "Design",
        "Analyze",
        "Optimize"
    ])

    with tab_design:
        render_design_tab(project)

    with tab_analyze:
        render_analyze_tab(project)

    with tab_optimize:
        render_optimize_tab(project)

if __name__ == "__main__":
    main()

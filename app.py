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
    st.divider()
    st.subheader("📐 Generate Layouts")
    st.write("Generate candidate layouts using deterministic algorithms for the current qubits in the layout.")
    
    if st.button("Generate Candidate Layouts"):
        if len(project.qubits) < 2:
            st.warning("Please add at least 2 qubits to generate layouts.")
        else:
            candidates = []
            N = len(project.qubits)
            import math
            import copy
            
            # 1. Grid
            grid_q = copy.deepcopy(project.qubits)
            cols = math.ceil(math.sqrt(N))
            spacing = project.constraints.min_qubit_spacing_um * 1.5
            for i, q in enumerate(grid_q):
                q.x_um = float((i % cols) * spacing + 10.0)
                q.y_um = float((i // cols) * spacing + 10.0)
            candidates.append(("Grid", grid_q))
            
            # 2. Ring
            ring_q = copy.deepcopy(project.qubits)
            radius = spacing * N / (2 * math.pi)
            radius = max(radius, spacing)
            for i, q in enumerate(ring_q):
                angle = 2 * math.pi * i / N
                q.x_um = float(radius * math.cos(angle) + radius + 10.0)
                q.y_um = float(radius * math.sin(angle) + radius + 10.0)
            candidates.append(("Ring", ring_q))
            
            # 3. Line (Nearest-Neighbor)
            line_q = copy.deepcopy(project.qubits)
            for i, q in enumerate(line_q):
                q.x_um = float(i * spacing + 10.0)
                q.y_um = 10.0
            candidates.append(("Line (Nearest-Neighbor)", line_q))
            
            best_score = -1000000.0
            best_name = ""
            best_qubits = None
            
            st.markdown("### Candidate Evaluation")
            c_cols = st.columns(3)
            
            for idx, (name, c_qubits) in enumerate(candidates):
                c_lqs = compute_lqs(c_qubits, project.connections)
                c_drc = validate_layout(c_qubits, project.constraints, project.connections, project.chip_width_um, project.chip_height_um)
                c_risks = build_risk_results(c_qubits, project.connections)
                c_worst_risk = max((r.objective_penalty for r in c_risks), default=0.0)
                
                # Scoring rule: 1. Prefer zero hard failures 2. Higher LQS 3. Lower worst-pair risk
                c_fail = len(c_drc.violations)
                score = - (c_fail * 1000.0) + c_lqs - c_worst_risk
                if score > best_score or best_name == "":
                    best_score = score
                    best_name = name
                    best_qubits = c_qubits
                    
                with c_cols[idx]:
                    st.markdown(f"**{name}**")
                    st.write(f"**LQS:** {c_lqs:.1f}")
                    st.write(f"**Worst Risk:** {c_worst_risk:.2f}")
                    drc_status = "🔴 FAIL" if c_fail > 0 else ("🟡 WARN" if len(c_drc.warnings) > 0 else "🟢 PASS")
                    st.write(f"**DRC:** {drc_status}")
                    
            st.success(f"🏆 **BEST CANDIDATE:** {best_name}")
            
            project_copy = copy.deepcopy(project)
            project_copy.qubits = best_qubits
            st.session_state["layout_gen_candidate"] = project_copy

    if "layout_gen_candidate" in st.session_state:
        st.info("A candidate layout is pending. Apply it to overwrite the current coordinates.")
        if st.button("✅ Apply Generated Layout", type="primary"):
            _set_project(st.session_state["layout_gen_candidate"])
            del st.session_state["layout_gen_candidate"]
            st.session_state["editor_version"] += 1
            st.rerun()

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
    lqs = compute_lqs(project.qubits, project.connections)
    
    if drc.violations:
        drc_status = "🔴 FAIL"
    elif drc.warnings:
        drc_status = "🟡 WARN"
    else:
        drc_status = "🟢 PASS"
        
    worst_penalty = max((r.objective_penalty for r in risk_results), default=0.0)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Qubits", len(project.qubits))
    col2.metric("LQS Score", f"{lqs:.1f} / 100")
    col3.metric("Q-DRC Status", drc_status)
    if not risk_results:
        col4.metric("Worst Pair Risk", "0.00")
    elif worst_penalty > 0.65:
        col4.metric("Worst Pair Risk", f"{worst_penalty:.3f}", delta="Critical (>0.65)", delta_color="inverse")
    else:
        col4.metric("Worst Pair Risk", f"{worst_penalty:.3f}")
    
    st.divider()

    st.markdown("### Risk Visibility Controls")
    c1, c2, c3, c4 = st.columns(4)
    show_intended = c1.checkbox("Intended", value=True)
    show_high = c2.checkbox("High Risk", value=True)
    show_medium = c3.checkbox("Medium Risk", value=True)
    show_low = c4.checkbox("Low Risk", value=False)
    
    fig = _build_canvas(project, show_intended, show_high, show_medium, show_low)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### Visual Q-DRC Dashboard")
    if len(project.qubits) < 2:
        st.info("No checks available. Add at least two qubits to begin pairwise analysis.")
    else:
        if not drc.violations and not drc.warnings:
            st.success("🟢 **Q-DRC PASS** — All layout constraints satisfied.")
        else:
            if drc.violations:
                st.error(f"🔴 **FAIL — {len(drc.violations)} violation(s)**")
                for v in drc.violations:
                    st.write(f"- ❌ {v}")
            if drc.warnings:
                st.warning(f"🟡 **WARN — {len(drc.warnings)} warning(s)**")
                for w in drc.warnings:
                    st.write(f"- ⚠ {w}")

    st.divider()
    st.markdown("### 🔍 Qubit Inspector")
    if not project.qubits:
        st.info("No qubits available. Add a qubit in the Design tab to inspect it.")
    else:
        q_ids = [q.id for q in project.qubits]
        selected_q_id = st.selectbox("Select Qubit ID", options=q_ids, key="inspect_q_id_analyze")
        if selected_q_id:
            q = next((q for q in project.qubits if q.id == selected_q_id), None)
            if q:
                import math
                nearest_id = None
                nearest_dist = float('inf')
                for other_q in project.qubits:
                    if other_q.id != q.id:
                        dist = math.hypot(q.x_um - other_q.x_um, q.y_um - other_q.y_um)
                        if dist < nearest_dist:
                            nearest_dist = dist
                            nearest_id = other_q.id
                
                nearest_str = f"{nearest_id} ({nearest_dist:.1f} µm)" if nearest_id else "None"
                
                q_risks = [r for r in risk_results if r.source_qubit_id == q.id or r.target_qubit_id == q.id]
                
                if q_risks:
                    worst = max(q_risks, key=lambda r: r.objective_penalty)
                    icon = "🔴" if worst.severity == "HIGH" else "🟡" if worst.severity == "MEDIUM" else "🟢"
                    worst_str = f"{icon} {worst.severity} (Penalty: {worst.objective_penalty:.3f}, Pair: {worst.source_qubit_id} ↔ {worst.target_qubit_id})"
                else:
                    worst_str = "🟢 LOW / None"
                    
                st.info(
                    f"**QUBIT INSPECTOR — {q.id}**\n\n"
                    f"• **Position:** X: {q.x_um:.1f} µm, Y: {q.y_um:.1f} µm\n\n"
                    f"• **Frequency:** {q.frequency_mhz:,.0f} MHz\n\n"
                    f"• **Nearest Neighbor:** {nearest_str}\n\n"
                    f"• **Worst Pair Risk:** {worst_str}\n\n"
                    f"• **Recommended Mitigation:** Increase spatial or frequency separation, then re-evaluate the layout."
                )

    st.divider()
    st.subheader("⚠ Pair Risk Analysis")
    
    if not risk_results:
        st.info("Not enough qubits to analyze risk.")
    else:
        options = [f"{rr.source_qubit_id} ↔ {rr.target_qubit_id}" for rr in risk_results]
        selected_pair = st.selectbox("Select a risk pair to inspect its detailed explanation", options=[""] + options)

        if selected_pair:
            for rr in risk_results:
                if f"{rr.source_qubit_id} ↔ {rr.target_qubit_id}" == selected_pair:
                    explanation = explain_risk(rr)
                    icon = "🔴" if rr.severity == "HIGH" else "🟡" if rr.severity == "MEDIUM" else "🟢"
                    
                    st.markdown(f"### {selected_pair}\n**{icon} {rr.severity} RISK**")
                    
                    c_a, c_b = st.columns(2)
                    c_a.markdown(f"**Distance**\n{rr.distance_um:.1f} µm")
                    c_b.markdown(f"**Frequency Delta**\n{rr.frequency_delta_mhz:.1f} MHz")
                    
                    c_c, c_d = st.columns(2)
                    interaction = "Intended" if rr.interaction_weight > 0 else "Unintended"
                    c_c.markdown(f"**Interaction**\n{interaction}")
                    c_d.markdown(f"**Objective Penalty**\n{rr.objective_penalty:.3f}")
                    
                    st.markdown("**Primary Cause**")
                    for reason in explanation["reasons"]:
                        st.markdown(f"- {reason}")
                        
                    st.info("**Recommended proxy-level mitigation**\n• Increase physical separation between the qubits\n• OR increase their frequency separation\n• Re-run Q-DRC after modification")
                    break

    st.divider()
    st.subheader("🤖 Q-Layout Copilot")
    st.caption("*AI-Assisted / Heuristic Recommendation*")
    
    if not project.qubits:
        st.info("Copilot needs at least one qubit to provide recommendations.")
    else:
        worst_pair_str = "None"
        primary_issue = "No critical issues detected."
        recommendation = "Layout looks structurally sound. Consider running Auto-Optimize to improve LQS further."
        
        if drc.violations:
            primary_issue = "Hard DRC violations present (e.g. minimum spacing or boundary breaches)."
            recommendation = "Resolve hard DRC failures first. Move overlapping qubits apart or expand the chip boundary."
        elif risk_results:
            worst = max(risk_results, key=lambda r: r.objective_penalty)
            worst_pair_str = f"{worst.source_qubit_id} ↔ {worst.target_qubit_id}"
            if worst.severity == "HIGH":
                primary_issue = f"High coupling risk detected on pair {worst_pair_str}."
                if worst.distance_um < project.constraints.min_qubit_spacing_um * 2:
                    recommendation = "Increase spatial separation first. If connectivity must be preserved, review frequency allocation as a secondary mitigation."
                else:
                    recommendation = "Spatial separation is adequate, but frequency collision is causing high spectral risk. Shift the operating frequency of one of the qubits."
            elif worst.severity == "MEDIUM":
                primary_issue = f"Moderate coupling risk detected on pair {worst_pair_str}."
                recommendation = "Review frequency allocation or slightly increase spatial separation to improve safety margins."
        
        st.markdown(f"**Highest-Risk Target:** {worst_pair_str}")
        st.markdown(f"**Primary Contributor:**\n{primary_issue}")
        st.info(f"**Suggested proxy-level action:**\n{recommendation}")

def render_optimize_tab(project):
    st.subheader("⚡ Automatic Layout Optimization")
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
        st.subheader("Optimization Result")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Starting LQS", f"{result.lqs_before:.1f}")
        c2.metric("Candidate LQS", f"{result.lqs_after:.1f}")
        improvement = result.lqs_after - result.lqs_before
        c3.metric("Improvement", f"+{improvement:.1f}" if improvement > 0 else f"{improvement:.1f}")
        
        st.markdown("### 📈 LQS Convergence")
        if result.movements:
            x_vals = [0] + [m["iteration"] for m in result.movements]
            y_vals = [result.lqs_before] + [m["lqs_after"] for m in result.movements]
            
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines+markers",
                    name="LQS",
                    line=dict(color=COLORS["intended"], width=2),
                    marker=dict(size=6),
                )
            )
            fig.update_layout(
                xaxis_title="Iteration (Accepted Improvement Steps)",
                yaxis_title="Layout Quality Score (LQS)",
                paper_bgcolor=COLORS["bg"],
                plot_bgcolor=COLORS["bg"],
                font=dict(color=COLORS["text"]),
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Note: Chart plots recorded LQS progression across accepted improvement steps from Iteration 0.")
        else:
            st.info("Optimization history is unavailable for this run.")
            
        st.subheader("📊 Optimization Telemetry")
        
        reason_map = {
            "patience_exhausted": "No further LQS-improving legal moves found.",
            "no_movable_qubits": "No movable qubits were available.",
            "max_iterations": "Maximum optimization iterations reached.",
            "no_improvement": "No further LQS improvement was found.",
            "already_optimal": "Layout was already optimal under the current constraints.",
        }
        human_reason = reason_map.get(result.stopped_reason, f"Optimization stopped: {result.stopped_reason}")
        
        imp_str = f"+{improvement:.1f}" if improvement > 0 else f"{improvement:.1f}"
        st.info(
            f"**Starting LQS:** {result.lqs_before:.1f}\n\n"
            f"**Candidate LQS:** {result.lqs_after:.1f}\n\n"
            f"**Improvement:** {imp_str}\n\n"
            f"**Total Iterations:** {result.iterations}\n\n"
            f"**Accepted Moves:** {len(result.movements)}\n\n"
            f"**Stopped Reason:** {human_reason}"
        )
                
        st.divider()
        st.markdown("### Review Candidate Layout")
        c_apply, c_discard = st.columns(2)
        if c_apply.button("✅ Apply Candidate Layout", type="primary", use_container_width=True):
            _set_project(candidate)
            st.session_state["candidate_project"] = None
            st.session_state["last_optimization_result"] = None
            st.session_state["editor_version"] += 1
            st.success("Candidate layout applied successfully.")
            st.rerun()
            
        if c_discard.button("❌ Discard", use_container_width=True):
            st.session_state["candidate_project"] = None
            st.session_state["last_optimization_result"] = None
            st.rerun()
    else:
        st.info("Run Auto-Optimize to generate a candidate layout.")

def main():
    _init_session_state()
    project = _get_project()

    st.title("Q-Layout")
    st.caption(f"Project: {project.name}")

    # --- 3-TAB EDA ARCHITECTURE ---
    tab_design, tab_analyze, tab_optimize = st.tabs(
        ["🛠️ Design", "📊 Analyze", "🚀 Optimize"]
    )

    with tab_design:
        # Move the existing design-related UI into this tab.
        render_design_tab(project)

    with tab_analyze:
        # Move the existing analysis UI into this tab.
        render_analyze_tab(project)

    with tab_optimize:
        # Move the existing optimization UI into this tab.
        render_optimize_tab(project)

if __name__ == "__main__":
    main()

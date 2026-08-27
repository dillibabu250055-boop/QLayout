# Q-Layout

**Built for PEC Hacks 4.0 by Team Quantum Pioneers**

Q-Layout is a physics-informed pre-simulation EDA copilot for superconducting quantum-chip layouts. It bridges architectural design and expensive electromagnetic validation by coupling spatial proximity analysis with spectral frequency checks in real time.

---

## The Problem

Quantum hardware designers routinely waste hours running computationally expensive electromagnetic (EM) simulations—tools like Ansys HFSS and Qiskit Metal—only to discover basic spatial or frequency collisions late in the design cycle. These collisions introduce unintended crosstalk, routing penalties, and fabrication failures that are expensive to fix once a layout is committed to silicon.

## The Solution

Q-Layout replaces guesswork with a coupled spatial-spectral risk engine. Every qubit pair is evaluated against normalized distance-dependent spatial risk and frequency-separation spectral risk, producing a single, explainable Layout Quality Score (LQS) from 0 to 100. The built-in Quantum Design Rule Checker (Q-DRC) enforces hard constraints—boundary clearance, minimum qubit spacing, and frequency-collision rules—so designers get immediate, actionable feedback before ever launching an EM solver. An integrated Hill-Climbing Auto-Optimizer then proposes concrete coordinate adjustments that strictly improve LQS while preserving all hard constraints.

---

## Key Features

- **Quantum Design Rule Checker (Q-DRC):** Hard-constraint validation for boundary clearance, minimum qubit spacing, and frequency collisions (with intended-pair bypass and warning logging).
- **Explainable Layout Quality Score (LQS):** A single 0–100 score backed by a physics-informed penalty model that distinguishes intended coupling (routing cost) from unintended crosstalk (spatial-spectral risk).
- **Hill-Climbing Auto-Optimizer:** Deterministic 8-direction hill-climbing optimizer that generates candidate moves per iteration, rejects any candidate violating Q-DRC, and accepts moves only when LQS strictly improves—with full before/after telemetry and explainability.
- **Interactive Chip Canvas:** Dark-mode Plotly visualization with hover tooltips, severity-colored risk edges, and intended-connection overlays.
- **Risk Explainability Panel:** Click any pair to see exactly why it was flagged—spatial proximity, spectral crowding, or routing inefficiency.
- **JSON Export:** Download the current layout as a portable JSON specification for downstream EM workflows.

---

## Local Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/dillibabu250055-boop/QLayout.git
cd qlayout
pip install -r requirements.txt
streamlit run app.py
```

The app launches at `http://localhost:8501` by default.

---

## How to Use the Demo

The default demo chip loads 8 qubits on a 100 µm × 100 µm canvas. On first load, you will see a **Q1–Q4 frequency collision** flagged in the Q-DRC Warnings panel: the two qubits are too close together given their near-identical frequencies.

1. Observe the initial **LQS** in the bottom status bar and the red/amber risk edges on the canvas.
2. Click **Auto-Optimize** in the bottom-right control panel.
3. Watch the Plotly canvas update as the optimizer moves Q4 to a new coordinate, eliminating the frequency-collision penalty.
4. Compare the **Before** and **After** LQS values to see the improvement.

All optimization steps are logged in the success message: iterations run, final LQS, and the stopping reason.

---

## Technology Stack

| Component | Technology |
|---|---|
| Frontend / UI | Streamlit |
| Numerical Computing | NumPy |
| Visualization | Plotly |
| Optimization | Custom Hill-Climbing heuristic (Python) |
| Data Persistence | JSON |

---

## License

This project was built for PEC Hacks 4.0. All rights reserved.

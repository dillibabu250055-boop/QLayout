# 03_APP_FLOW.md

## Primary User Journey

1. **Dashboard Start:** User launches application.
2. **Configure Chip:** User inputs chip dimensions and hard constraints.
3. **Add Qubits:** User places qubits manually or loads a default layout.
4. **Assign Frequencies:** User sets microwave frequencies manually in MHz.
5. **Define Connectivity:** User defines intended circuit couplings (edges).
6. **Analyze:** Engine computes distances and frequency separations.
7. **Q-DRC:** System checks hard rules (boundaries, minimum spacing, delta-f minimums).
8. **Risk Visualization:** High-risk pairs highlight in red on the canvas.
9. **Select Violation:** User selects a red edge.
10. **View Explanation:** UI explains exact distance, frequency delta, and risk classification.
11. **Auto-Optimize:** User clicks [Optimize Layout].
12. **Compare:** UI shows LQS before and after optimization.
13. **Export:** User downloads JSON/CSV layout specification for downstream EM workflows.

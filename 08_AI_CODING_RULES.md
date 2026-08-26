# 08_AI_CODING_RULES.md

## SOURCE OF TRUTH HIERARCHY
If two documents conflict, you must STOP and request clarification rather than inventing a resolution. The documents are authoritative in this exact order:
1. `06_OPTIMIZATION_SPEC.md` — physics and optimization mathematics
2. `05_BACKEND_SCHEMA.md` — data contracts
3. `01_PRD.md` — product requirements
4. `07_IMPLEMENTATION_PLAN.md` — implementation sequence

## RULES
1. Read `01_PRD.md` before implementing features.
2. Read `02_TRD.md` before selecting technologies.
3. Read `03_APP_FLOW.md` before modifying navigation.
4. Read `06_OPTIMIZATION_SPEC.md` before modifying physics or optimization logic.
5. Never invent physics formulas.
6. Never change the risk model or LQS bounds without explicit approval.
7. Never introduce a new dependency unless necessary.
8. Keep physics calculations separate from UI code.
9. Keep optimization logic separate from Streamlit code.
10. Strictly adhere to standard units (um for distance, MHz for frequency).
11. **Optimizer Invariant 1:** The optimizer must never accept a candidate that violates a hard Q-DRC constraint.
12. **Optimizer Invariant 2:** An optimization step may only be accepted if the objective improves according to the canonical LQS calculation.
13. **Optimizer Invariant 3:** After optimization, the complete Q-DRC and risk engine must be rerun.
14. Never hard-code demo results outside of the Phase 8 Demo Dataset.
15. Never claim proxy calculations are EM simulation results.
16. Never describe proxy risk values as measured, simulated, or exact electromagnetic coupling.

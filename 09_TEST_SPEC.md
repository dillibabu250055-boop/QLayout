# 09_TEST_SPEC.md

## 1. Mathematical Invariants
- **Spatial risk:** $0 \le S_{ij} \le 1$
- **Spectral risk:** $0 < F_{ij} \le 1$
- **Base risk:** $0 \le R_{ij} \le 1$
- **Objective penalty:** $0 \le P_{ij} \le 1$
- **LQS:** $0 \le LQS \le 100$

## 2. Symmetry Tests
- `distance(i,j) == distance(j,i)`
- `risk(i,j) == risk(j,i)`

## 3. Boundary Tests
- Qubit inside boundary → PASS
- Qubit outside boundary → FAIL

## 4. Spacing Tests
- `distance >= minimum_spacing` → PASS
- `distance < minimum_spacing` → FAIL

## 5. Frequency Tests (Q-DRC)
- Unintended pair + close + insufficient $\Delta f$ → FAIL
- Unintended pair + close + sufficient $\Delta f$ → PASS
- Intended pair + close + insufficient $\Delta f$ → PASS (Log as Warning)

## 6. Connectivity Tests
- $I_{ij} = 1.0$ → Intended
- $I_{ij} = 0.0$ → Unintended

## 7. Optimization Tests
- **Accepted optimization step:** $LQS_{new} > LQS_{old}$
- **Complete optimization run:** $LQS_{final} \ge LQS_{initial}$
- Hard violations never increase after an accepted optimization step.
- Optimizer is perfectly deterministic with seed 42.
- **Edge Case - No movable qubits:** Optimizer must terminate safely, layout must remain unchanged, $LQS_{final} == LQS_{initial}$.
- **Edge Case - All candidates invalid:** If all candidate moves violate hard constraints, optimizer must terminate safely, and no invalid layout may be returned.

import random
from typing import List, Dict, Optional

from models.schema import Qubit, Connection, ChipConstraints, OptimizationResult
from core.q_drc import validate_layout
from core.scoring import compute_lqs
from core.physics_engine import build_risk_results


MAX_ITERATIONS = 500
PATIENCE = 50
STEP_SIZE_UM = 5.0
RANDOM_SEED = 42

DIRECTIONS = [
    (0.0, 1.0),
    (0.0, -1.0),
    (1.0, 0.0),
    (-1.0, 0.0),
    (1.0, 1.0),
    (-1.0, 1.0),
    (1.0, -1.0),
    (-1.0, -1.0),
]


def _copy_qubits(qubits: List[Qubit]) -> List[Qubit]:
    return [Qubit(q.id, q.x_um, q.y_um, q.frequency_mhz, q.movable) for q in qubits]


def _count_violations(drc_result) -> int:
    return len(drc_result.violations)


def _select_target_qubit(
    qubits: List[Qubit],
    connections: List[Connection],
    constraints: ChipConstraints,
    chip_width_um: float,
    chip_height_um: float,
) -> Optional[Qubit]:
    if not connections:
        return None

    drc_result = validate_layout(qubits, constraints, connections, chip_width_um, chip_height_um)

    violating_qubit_ids = set()
    if drc_result.violations:
        for v in drc_result.violations:
            if "qubit_id" in v:
                violating_qubit_ids.add(v["qubit_id"])
            if "qubit_a" in v:
                violating_qubit_ids.add(v["qubit_a"])
                violating_qubit_ids.add(v["qubit_b"])

    risk_results = build_risk_results(qubits, connections)
    if not risk_results:
        return None

    best_penalty = -1.0
    best_qubit_id = None
    best_target_id = None

    for rr in risk_results:
        candidate_a = rr.source_qubit_id
        candidate_b = rr.target_qubit_id

        if drc_result.violations:
            if candidate_a not in violating_qubit_ids and candidate_b not in violating_qubit_ids:
                continue

        if rr.objective_penalty > best_penalty:
            best_penalty = rr.objective_penalty
            best_qubit_id = candidate_a
            best_target_id = candidate_b

    if best_qubit_id is None:
        return None

    for q in qubits:
        if q.id == best_qubit_id and q.movable:
            return q
    for q in qubits:
        if q.id == best_target_id and q.movable:
            return q

    return None


def optimize_layout(
    qubits: List[Qubit],
    connections: List[Connection],
    constraints: ChipConstraints,
    chip_width_um: float,
    chip_height_um: float,
) -> OptimizationResult:
    random.seed(RANDOM_SEED)

    current_qubits = _copy_qubits(qubits)

    before_lqs = compute_lqs(current_qubits, connections)
    before_drc = validate_layout(current_qubits, constraints, connections, chip_width_um, chip_height_um)
    violations_before = _count_violations(before_drc)

    movements: List[Dict] = []
    iterations = 0
    patience_counter = 0
    stopped_reason = "max_iterations"

    for iteration in range(MAX_ITERATIONS):
        iterations += 1

        target = _select_target_qubit(current_qubits, connections, constraints, chip_width_um, chip_height_um)
        if target is None:
            stopped_reason = "no_movable_qubits"
            break

        target_idx = next(i for i, q in enumerate(current_qubits) if q.id == target.id)
        current_lqs = compute_lqs(current_qubits, connections)
        current_violations = _count_violations(
            validate_layout(current_qubits, constraints, connections, chip_width_um, chip_height_um)
        )

        best_candidate = None
        best_lqs = current_lqs

        for dx, dy in DIRECTIONS:
            new_x = target.x_um + dx * STEP_SIZE_UM
            new_y = target.y_um + dy * STEP_SIZE_UM

            candidate_qubits = _copy_qubits(current_qubits)
            candidate_qubits[target_idx] = Qubit(
                target.id, new_x, new_y, target.frequency_mhz, target.movable
            )

            drc = validate_layout(candidate_qubits, constraints, connections, chip_width_um, chip_height_um)
            if not drc.passed:
                continue
            lqs = compute_lqs(candidate_qubits, connections)
            if lqs > best_lqs:
                best_lqs = lqs
                best_candidate = candidate_qubits

        if best_candidate is None:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                stopped_reason = "patience_exhausted"
                break
            continue

        new_target = best_candidate[target_idx]
        violations_after = _count_violations(
            validate_layout(best_candidate, constraints, connections, chip_width_um, chip_height_um)
        )
        movements.append({
            "iteration": iteration + 1,
            "qubit_id": target.id,
            "from_x": target.x_um,
            "from_y": target.y_um,
            "to_x": new_target.x_um,
            "to_y": new_target.y_um,
            "lqs_before": current_lqs,
            "lqs_after": best_lqs,
            "violations_before": current_violations,
            "violations_after": violations_after,
        })

        current_qubits = best_candidate
        patience_counter = 0

    after_lqs = compute_lqs(current_qubits, connections)
    after_drc = validate_layout(current_qubits, constraints, connections, chip_width_um, chip_height_um)
    violations_after = _count_violations(after_drc)

    if before_lqs > 0:
        improvement_percent = ((after_lqs - before_lqs) / before_lqs) * 100.0
    else:
        improvement_percent = 0.0

    return OptimizationResult(
        before_lqs=before_lqs,
        after_lqs=after_lqs,
        improvement_percent=improvement_percent,
        iterations=iterations,
        movements=movements,
        violations_before=violations_before,
        violations_after=violations_after,
        objective_before=before_lqs,
        objective_after=after_lqs,
        stopped_reason=stopped_reason,
    )

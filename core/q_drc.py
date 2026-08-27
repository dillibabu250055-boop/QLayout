from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple
import numpy as np

from core.scoring import get_severity_from_penalty
from models.schema import Qubit, Connection, ChipConstraints


@dataclass
class DRCCheckResult:
    passed: bool
    violations: List[Dict] = field(default_factory=list)
    warnings: List[Dict] = field(default_factory=list)


def _build_intended_pairs(connections: List[Connection]) -> Set[Tuple[str, str]]:
    pairs: Set[Tuple[str, str]] = set()
    for conn in connections:
        if conn.interaction_weight <= 0.0:
            continue
        a, b = conn.source_qubit_id, conn.target_qubit_id
        pair = (a, b) if a < b else (b, a)
        pairs.add(pair)
    return pairs


def check_boundary(
    qubits: List[Qubit],
    constraints: ChipConstraints,
    chip_width_um: float,
    chip_height_um: float
) -> List[Dict]:
    violations = []
    min_clear = constraints.min_boundary_clearance_um
    for q in qubits:
        if q.x_um < min_clear or q.x_um > (chip_width_um - min_clear):
            violations.append({
                "qubit_id": q.id,
                "rule": "boundary",
                "detail": f"x={q.x_um} outside [{min_clear}, {chip_width_um - min_clear}]"
            })
        if q.y_um < min_clear or q.y_um > (chip_height_um - min_clear):
            violations.append({
                "qubit_id": q.id,
                "rule": "boundary",
                "detail": f"y={q.y_um} outside [{min_clear}, {chip_height_um - min_clear}]"
            })
    return violations


def check_spacing(qubits: List[Qubit], constraints: ChipConstraints) -> List[Dict]:
    violations = []
    min_spacing = constraints.min_qubit_spacing_um
    n = len(qubits)
    for i in range(n):
        for j in range(i + 1, n):
            d_ij = np.sqrt(
                (qubits[i].x_um - qubits[j].x_um) ** 2 +
                (qubits[i].y_um - qubits[j].y_um) ** 2
            )
            if d_ij < min_spacing:
                violations.append({
                    "qubit_a": qubits[i].id,
                    "qubit_b": qubits[j].id,
                    "rule": "spacing",
                    "distance_um": float(d_ij),
                    "min_spacing_um": min_spacing,
                    "detail": f"distance={d_ij:.2f} < {min_spacing}"
                })
    return violations


def check_frequency_collisions(
    qubits: List[Qubit],
    constraints: ChipConstraints,
    connections: List[Connection]
) -> Tuple[List[Dict], List[Dict]]:
    violations = []
    warnings = []
    freq_dist = constraints.frequency_check_distance_um
    min_sep = constraints.min_frequency_separation_mhz
    intended = _build_intended_pairs(connections)
    id_to_qubit = {q.id: q for q in qubits}
    n = len(qubits)

    for i in range(n):
        for j in range(i + 1, n):
            qi = qubits[i]
            qj = qubits[j]
            d_ij = np.sqrt((qi.x_um - qj.x_um) ** 2 + (qi.y_um - qj.y_um) ** 2)
            if d_ij >= freq_dist:
                continue
            delta_f = abs(qi.frequency_mhz - qj.frequency_mhz)
            if delta_f >= min_sep:
                continue
            a, b = qi.id, qj.id
            pair = (a, b) if a < b else (b, a)
            entry = {
                "qubit_a": a,
                "qubit_b": b,
                "distance_um": float(d_ij),
                "frequency_delta_mhz": float(delta_f),
                "min_separation_mhz": min_sep,
                "rule": "frequency",
                "detail": f"distance={d_ij:.2f}, delta_f={delta_f:.2f} < {min_sep}"
            }
            if pair in intended:
                warnings.append(entry)
            else:
                violations.append(entry)
    return violations, warnings


def validate_layout(
    qubits: List[Qubit],
    constraints: ChipConstraints,
    connections: List[Connection],
    chip_width_um: float,
    chip_height_um: float
) -> DRCCheckResult:
    boundary_violations = check_boundary(qubits, constraints, chip_width_um, chip_height_um)
    spacing_violations = check_spacing(qubits, constraints)
    freq_violations, freq_warnings = check_frequency_collisions(qubits, constraints, connections)
    all_violations = boundary_violations + spacing_violations + freq_violations
    return DRCCheckResult(
        passed=len(all_violations) == 0,
        violations=all_violations,
        warnings=freq_warnings
    )

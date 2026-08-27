from dataclasses import dataclass
from typing import List, Dict


@dataclass
class ChipConstraints:
    min_qubit_spacing_um: float
    min_frequency_separation_mhz: float
    frequency_check_distance_um: float
    min_boundary_clearance_um: float

    def __post_init__(self):
        if self.min_qubit_spacing_um < 0:
            raise ValueError("min_qubit_spacing_um must be >= 0")
        if self.min_frequency_separation_mhz < 0:
            raise ValueError("min_frequency_separation_mhz must be >= 0")
        if self.frequency_check_distance_um <= 0:
            raise ValueError("frequency_check_distance_um must be > 0")
        if self.min_boundary_clearance_um < 0:
            raise ValueError("min_boundary_clearance_um must be >= 0")


@dataclass
class Qubit:
    id: str
    x_um: float
    y_um: float
    frequency_mhz: float
    movable: bool

    def __post_init__(self):
        if self.x_um < 0:
            raise ValueError("Qubit x_um must be >= 0")
        if self.y_um < 0:
            raise ValueError("Qubit y_um must be >= 0")
        if self.frequency_mhz <= 0:
            raise ValueError("Qubit frequency_mhz must be > 0")


@dataclass
class Connection:
    source_qubit_id: str
    target_qubit_id: str
    interaction_weight: float
    gate_count: int

    def __post_init__(self):
        if not 0.0 <= float(self.interaction_weight) <= 1.0:
            raise ValueError("interaction_weight must be between 0.0 and 1.0")


@dataclass
class RiskResult:
    source_qubit_id: str
    target_qubit_id: str
    distance_um: float
    frequency_delta_mhz: float
    spatial_risk: float
    spectral_risk: float
    base_interaction_risk: float
    interaction_weight: float
    unintended_risk: float
    routing_cost: float
    objective_penalty: float
    severity: str


@dataclass
class OptimizationResult:
    lqs_before: float
    lqs_after: float
    improvement_percent: float
    iterations: int
    movements: List[Dict]
    violations_before: int
    violations_after: int
    stopped_reason: str


@dataclass
class Project:
    id: str
    name: str
    chip_width_um: float
    chip_height_um: float
    constraints: ChipConstraints
    qubits: List[Qubit]
    connections: List[Connection]

    def __post_init__(self):
        qubit_ids = [q.id for q in self.qubits]
        if any(not str(qid).strip() for qid in qubit_ids):
            raise ValueError("Qubit IDs must be non-empty")
        if len(qubit_ids) != len(set(qubit_ids)):
            raise ValueError("Qubit IDs must be unique")

        valid_ids = set(qubit_ids)
        seen_pairs = set()
        for conn in self.connections:
            if conn.source_qubit_id not in valid_ids or conn.target_qubit_id not in valid_ids:
                raise ValueError("Connection references unknown qubit IDs")
            if conn.source_qubit_id == conn.target_qubit_id:
                raise ValueError("A qubit cannot connect to itself")
            pair = frozenset({conn.source_qubit_id, conn.target_qubit_id})
            if pair in seen_pairs:
                raise ValueError("Duplicate logical connection is not allowed")
            seen_pairs.add(pair)

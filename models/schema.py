from dataclasses import dataclass
from typing import List, Dict


@dataclass
class ChipConstraints:
    min_qubit_spacing_um: float
    min_frequency_separation_mhz: float
    frequency_check_distance_um: float
    min_boundary_clearance_um: float


@dataclass
class Qubit:
    id: str
    x_um: float
    y_um: float
    frequency_mhz: float
    movable: bool


@dataclass
class Connection:
    source_qubit_id: str
    target_qubit_id: str
    interaction_weight: float
    gate_count: int


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
    before_lqs: float
    after_lqs: float
    improvement_percent: float
    iterations: int
    movements: List[Dict]
    violations_before: int
    violations_after: int
    objective_before: float
    objective_after: float
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

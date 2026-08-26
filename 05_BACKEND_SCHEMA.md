# 05_BACKEND_SCHEMA.md

## Data Models

**Project**
- id: str
- name: str
- chip_width_um: float
- chip_height_um: float
- constraints: ChipConstraints
- qubits: list[Qubit]
- connections: list[Connection]

**ChipConstraints**
- min_qubit_spacing_um: float
- min_frequency_separation_mhz: float
- frequency_check_distance_um: float
- min_boundary_clearance_um: float

**Qubit**
- id: str
- x_um: float
- y_um: float
- frequency_mhz: float
- movable: bool

**Connection**
- source_qubit_id: str
- target_qubit_id: str
- interaction_weight: float   # I_ij bounded [0.0, 1.0]
- gate_count: int

**RiskResult**
- source_qubit_id: str
- target_qubit_id: str
- distance_um: float
- frequency_delta_mhz: float
- spatial_risk: float         # Normalized [0, 1]
- spectral_risk: float        # Normalized [0, 1]
- base_interaction_risk: float # Normalized [0, 1]
- interaction_weight: float   # Normalized [0, 1]
- unintended_risk: float      # Normalized [0, 1]
- routing_cost: float         # Normalized [0, 1]
- objective_penalty: float    # P_ij
- severity: str               # 'HIGH' (>0.65), 'MEDIUM' (0.3-0.65), 'LOW' (<0.3)

**OptimizationResult**
- before_lqs: float
- after_lqs: float
- improvement_percent: float
- iterations: int
- movements: list[dict]
- violations_before: int
- violations_after: int
- objective_before: float
- objective_after: float
- stopped_reason: str

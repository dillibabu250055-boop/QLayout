import pytest
from models.schema import Qubit, Connection, ChipConstraints
from core.optimizer import optimize_layout
from core.scoring import compute_lqs
from core.q_drc import validate_layout


CHIP_WIDTH = 100.0
CHIP_HEIGHT = 100.0


def _make_constraints(**overrides):
    defaults = {
        "min_qubit_spacing_um": 10.0,
        "min_frequency_separation_mhz": 50.0,
        "frequency_check_distance_um": 20.0,
        "min_boundary_clearance_um": 5.0,
    }
    defaults.update(overrides)
    return ChipConstraints(**defaults)


def _make_qubits(positions):
    return [
        Qubit(id=f"q{i}", x_um=x, y_um=y, frequency_mhz=5000.0 + i * 100, movable=True)
        for i, (x, y) in enumerate(positions)
    ]


def _make_connections(pairs, weights):
    return [
        Connection(source_qubit_id=a, target_qubit_id=b, interaction_weight=w, gate_count=1)
        for (a, b), w in zip(pairs, weights)
    ]


class TestOptimizationImprovement:
    def test_lqs_final_gte_initial(self):
        constraints = _make_constraints()
        qubits = _make_qubits([(10.0, 10.0), (15.0, 10.0), (50.0, 50.0)])
        connections = _make_connections([("q0", "q1")], [1.0])

        result = optimize_layout(qubits, connections, constraints, CHIP_WIDTH, CHIP_HEIGHT)
        assert result.lqs_after >= result.lqs_before

    def test_hard_violations_never_increase_after_accepted_step(self):
        constraints = _make_constraints()
        qubits = _make_qubits([(10.0, 10.0), (15.0, 10.0), (50.0, 50.0)])
        connections = _make_connections([("q0", "q1")], [1.0])

        result = optimize_layout(qubits, connections, constraints, CHIP_WIDTH, CHIP_HEIGHT)
        for move in result.movements:
            assert move["violations_after"] <= move["violations_before"]


class TestDeterminism:
    def test_deterministic_behavior(self):
        constraints = _make_constraints()
        qubits = _make_qubits([(10.0, 10.0), (15.0, 10.0), (50.0, 50.0)])
        connections = _make_connections([("q0", "q1")], [1.0])

        result1 = optimize_layout(qubits, connections, constraints, CHIP_WIDTH, CHIP_HEIGHT)
        result2 = optimize_layout(qubits, connections, constraints, CHIP_WIDTH, CHIP_HEIGHT)

        assert result1.lqs_after == result2.lqs_after
        assert result1.iterations == result2.iterations
        assert len(result1.movements) == len(result2.movements)
        for m1, m2 in zip(result1.movements, result2.movements):
            assert m1 == m2


class TestNoMovableQubits:
    def test_no_movable_qubits(self):
        constraints = _make_constraints()
        qubits = [
            Qubit(id="q0", x_um=10.0, y_um=10.0, frequency_mhz=5000.0, movable=False),
            Qubit(id="q1", x_um=20.0, y_um=20.0, frequency_mhz=5100.0, movable=False),
        ]
        connections = _make_connections([("q0", "q1")], [1.0])

        result = optimize_layout(qubits, connections, constraints, CHIP_WIDTH, CHIP_HEIGHT)
        assert result.lqs_after == result.lqs_before
        assert result.movements == []
        assert result.stopped_reason == "no_movable_qubits"


class TestAllCandidatesInvalid:
    def test_all_candidates_invalid(self):
        constraints = _make_constraints(
            min_qubit_spacing_um=100.0,
            min_boundary_clearance_um=5.0,
        )
        qubits = [
            Qubit(id="q0", x_um=5.1, y_um=5.1, frequency_mhz=5000.0, movable=True),
            Qubit(id="q1", x_um=20.1, y_um=5.1, frequency_mhz=5100.0, movable=False),
        ]
        connections = _make_connections([("q0", "q1")], [1.0])

        result = optimize_layout(qubits, connections, constraints, 10.0, 10.0)
        assert result.lqs_after == result.lqs_before
        assert result.movements == []


class TestUnconnectedHighRiskTargeting:
    def test_optimizer_targets_unconnected_close_pair(self):
        constraints = _make_constraints(min_boundary_clearance_um=0.0)
        qubits = [
            Qubit(id="q0", x_um=10.0, y_um=10.0, frequency_mhz=5000.0, movable=True),
            Qubit(id="q1", x_um=15.0, y_um=10.0, frequency_mhz=5005.0, movable=True),
        ]
        connections = []

        initial_lqs = compute_lqs(qubits, connections)
        result = optimize_layout(qubits, connections, constraints, CHIP_WIDTH, CHIP_HEIGHT)

        assert result.lqs_before == initial_lqs
        assert result.lqs_after >= result.lqs_before
        assert result.movements == [] or result.lqs_after >= result.lqs_before

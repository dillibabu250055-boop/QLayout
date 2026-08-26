import pytest
from models.schema import Qubit, Connection, ChipConstraints, RiskResult, Project
from core.q_drc import (
    validate_layout,
    check_boundary,
    check_spacing,
    check_frequency_collisions,
    severity_from_penalty,
)
from core.scoring import compute_lqs, explain_risk, get_severity_from_penalty


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


class TestBoundary:
    def test_inside_boundary_pass(self):
        constraints = _make_constraints()
        qubits = _make_qubits([(10.0, 10.0), (20.0, 20.0)])
        result = validate_layout(qubits, constraints, [], chip_width_um=CHIP_WIDTH, chip_height_um=CHIP_HEIGHT)
        assert result.passed is True

    def test_outside_boundary_fail(self):
        constraints = _make_constraints(min_boundary_clearance_um=5.0)
        qubits = _make_qubits([(3.0, 10.0), (20.0, 20.0)])
        result = validate_layout(qubits, constraints, [], chip_width_um=CHIP_WIDTH, chip_height_um=CHIP_HEIGHT)
        assert result.passed is False
        assert any(v["rule"] == "boundary" for v in result.violations)

    def test_on_boundary_edge_pass(self):
        constraints = _make_constraints(min_boundary_clearance_um=5.0)
        qubits = _make_qubits([(5.0, 5.0), (95.0, 95.0)])
        result = validate_layout(qubits, constraints, [], chip_width_um=CHIP_WIDTH, chip_height_um=CHIP_HEIGHT)
        assert result.passed is True

    def test_below_min_y_fail(self):
        constraints = _make_constraints(min_boundary_clearance_um=5.0)
        qubits = _make_qubits([(10.0, 3.0)])
        result = validate_layout(qubits, constraints, [], chip_width_um=CHIP_WIDTH, chip_height_um=CHIP_HEIGHT)
        assert result.passed is False


class TestSpacing:
    def test_distance_gte_minimum_pass(self):
        constraints = _make_constraints(min_qubit_spacing_um=10.0, min_boundary_clearance_um=0.0)
        qubits = _make_qubits([(0.0, 0.0), (15.0, 0.0)])
        result = validate_layout(qubits, constraints, [], chip_width_um=CHIP_WIDTH, chip_height_um=CHIP_HEIGHT)
        assert result.passed is True

    def test_distance_lt_minimum_fail(self):
        constraints = _make_constraints(min_qubit_spacing_um=10.0, min_boundary_clearance_um=0.0)
        qubits = _make_qubits([(0.0, 0.0), (8.0, 0.0)])
        result = validate_layout(qubits, constraints, [], chip_width_um=CHIP_WIDTH, chip_height_um=CHIP_HEIGHT)
        assert result.passed is False
        assert any(v["rule"] == "spacing" for v in result.violations)

    def test_exact_minimum_spacing_pass(self):
        constraints = _make_constraints(min_qubit_spacing_um=10.0, min_boundary_clearance_um=0.0)
        qubits = _make_qubits([(0.0, 0.0), (10.0, 0.0)])
        result = validate_layout(qubits, constraints, [], chip_width_um=CHIP_WIDTH, chip_height_um=CHIP_HEIGHT)
        assert result.passed is True

    def test_diagonal_spacing_fail(self):
        constraints = _make_constraints(min_qubit_spacing_um=10.0, min_boundary_clearance_um=0.0)
        qubits = _make_qubits([(0.0, 0.0), (7.0, 7.0)])
        result = validate_layout(qubits, constraints, [], chip_width_um=CHIP_WIDTH, chip_height_um=CHIP_HEIGHT)
        assert result.passed is False


class TestFrequency:
    def test_unintended_close_insufficient_freq_fail(self):
        constraints = _make_constraints(
            min_qubit_spacing_um=10.0,
            min_frequency_separation_mhz=50.0,
            frequency_check_distance_um=20.0,
            min_boundary_clearance_um=0.0,
        )
        qubits = _make_qubits([(0.0, 0.0), (15.0, 0.0)])
        qubits[1].frequency_mhz = 5010.0
        connections = _make_connections([("q0", "q1")], [0.0])
        result = validate_layout(qubits, constraints, connections, chip_width_um=CHIP_WIDTH, chip_height_um=CHIP_HEIGHT)
        assert result.passed is False
        assert any(v["rule"] == "frequency" for v in result.violations)

    def test_unintended_close_sufficient_freq_pass(self):
        constraints = _make_constraints(
            min_qubit_spacing_um=10.0,
            min_frequency_separation_mhz=50.0,
            frequency_check_distance_um=20.0,
            min_boundary_clearance_um=0.0,
        )
        qubits = _make_qubits([(0.0, 0.0), (15.0, 0.0)])
        qubits[1].frequency_mhz = 5100.0
        connections = _make_connections([("q0", "q1")], [0.0])
        result = validate_layout(qubits, constraints, connections, chip_width_um=CHIP_WIDTH, chip_height_um=CHIP_HEIGHT)
        assert result.passed is True

    def test_intended_close_insufficient_freq_pass_with_warning(self):
        constraints = _make_constraints(
            min_qubit_spacing_um=10.0,
            min_frequency_separation_mhz=50.0,
            frequency_check_distance_um=20.0,
            min_boundary_clearance_um=0.0,
        )
        qubits = _make_qubits([(0.0, 0.0), (15.0, 0.0)])
        qubits[1].frequency_mhz = 5010.0
        connections = _make_connections([("q0", "q1")], [1.0])
        result = validate_layout(qubits, constraints, connections, chip_width_um=CHIP_WIDTH, chip_height_um=CHIP_HEIGHT)
        assert result.passed is True
        assert any(w["rule"] == "frequency" for w in result.warnings)

    def test_far_pair_ignores_frequency(self):
        constraints = _make_constraints(
            min_qubit_spacing_um=10.0,
            min_frequency_separation_mhz=50.0,
            frequency_check_distance_um=20.0,
            min_boundary_clearance_um=0.0,
        )
        qubits = _make_qubits([(0.0, 0.0), (50.0, 0.0)])
        qubits[1].frequency_mhz = 5005.0
        connections = _make_connections([("q0", "q1")], [0.0])
        result = validate_layout(qubits, constraints, connections, chip_width_um=CHIP_WIDTH, chip_height_um=CHIP_HEIGHT)
        assert result.passed is True


class TestSeverityClassification:
    def test_low_severity(self):
        assert severity_from_penalty(0.0) == "LOW"
        assert severity_from_penalty(0.299999) == "LOW"
        assert severity_from_penalty(0.1) == "LOW"

    def test_medium_severity(self):
        assert severity_from_penalty(0.30) == "MEDIUM"
        assert severity_from_penalty(0.5) == "MEDIUM"
        assert severity_from_penalty(0.65) == "MEDIUM"

    def test_high_severity(self):
        assert severity_from_penalty(0.650001) == "HIGH"
        assert severity_from_penalty(0.66) == "HIGH"
        assert severity_from_penalty(1.0) == "HIGH"

    def test_canonical_get_severity_matches_q_drc(self):
        for value in [0.0, 0.299999, 0.30, 0.65, 0.650001, 1.0]:
            assert get_severity_from_penalty(value) == severity_from_penalty(value)


class TestQubitValidation:
    def test_negative_x_raises(self):
        with pytest.raises(ValueError, match="x_um"):
            Qubit(id="q0", x_um=-1.0, y_um=0.0, frequency_mhz=5000.0, movable=True)

    def test_negative_y_raises(self):
        with pytest.raises(ValueError, match="y_um"):
            Qubit(id="q0", x_um=0.0, y_um=-1.0, frequency_mhz=5000.0, movable=True)

    def test_zero_frequency_raises(self):
        with pytest.raises(ValueError, match="frequency_mhz"):
            Qubit(id="q0", x_um=0.0, y_um=0.0, frequency_mhz=0.0, movable=True)

    def test_negative_frequency_raises(self):
        with pytest.raises(ValueError, match="frequency_mhz"):
            Qubit(id="q0", x_um=0.0, y_um=0.0, frequency_mhz=-100.0, movable=True)


class TestProjectValidation:
    def test_empty_qubit_id_raises(self):
        q = Qubit(id="q0", x_um=0.0, y_um=0.0, frequency_mhz=5000.0, movable=True)
        with pytest.raises(ValueError, match="Qubit IDs"):
            Project(
                id="p1",
                name="demo",
                chip_width_um=100.0,
                chip_height_um=100.0,
                constraints=_make_constraints(),
                qubits=[Qubit(id="", x_um=1.0, y_um=1.0, frequency_mhz=5000.0, movable=True)],
                connections=[],
            )

    def test_duplicate_qubit_ids_raises(self):
        qubits = [
            Qubit(id="q0", x_um=0.0, y_um=0.0, frequency_mhz=5000.0, movable=True),
            Qubit(id="q0", x_um=2.0, y_um=2.0, frequency_mhz=5100.0, movable=True),
        ]
        with pytest.raises(ValueError, match="unique"):
            Project(
                id="p1",
                name="demo",
                chip_width_um=100.0,
                chip_height_um=100.0,
                constraints=_make_constraints(),
                qubits=qubits,
                connections=[],
            )

    def test_unknown_connection_endpoint_raises(self):
        qubits = [Qubit(id="q0", x_um=0.0, y_um=0.0, frequency_mhz=5000.0, movable=True)]
        with pytest.raises(ValueError, match="unknown qubit IDs"):
            Project(
                id="p1",
                name="demo",
                chip_width_um=100.0,
                chip_height_um=100.0,
                constraints=_make_constraints(),
                qubits=qubits,
                connections=[Connection(source_qubit_id="q0", target_qubit_id="q1", interaction_weight=1.0, gate_count=1)],
            )

    def test_self_connection_raises(self):
        qubits = [Qubit(id="q0", x_um=0.0, y_um=0.0, frequency_mhz=5000.0, movable=True)]
        with pytest.raises(ValueError, match="cannot connect to itself"):
            Project(
                id="p1",
                name="demo",
                chip_width_um=100.0,
                chip_height_um=100.0,
                constraints=_make_constraints(),
                qubits=qubits,
                connections=[Connection(source_qubit_id="q0", target_qubit_id="q0", interaction_weight=1.0, gate_count=1)],
            )

    def test_duplicate_connection_raises(self):
        qubits = [
            Qubit(id="q0", x_um=0.0, y_um=0.0, frequency_mhz=5000.0, movable=True),
            Qubit(id="q1", x_um=5.0, y_um=0.0, frequency_mhz=5100.0, movable=True),
        ]
        with pytest.raises(ValueError, match="Duplicate logical connection"):
            Project(
                id="p1",
                name="demo",
                chip_width_um=100.0,
                chip_height_um=100.0,
                constraints=_make_constraints(),
                qubits=qubits,
                connections=[
                    Connection(source_qubit_id="q0", target_qubit_id="q1", interaction_weight=1.0, gate_count=1),
                    Connection(source_qubit_id="q0", target_qubit_id="q1", interaction_weight=1.0, gate_count=1),
                ],
            )

    def test_reversed_connection_raises(self):
        qubits = [
            Qubit(id="q0", x_um=0.0, y_um=0.0, frequency_mhz=5000.0, movable=True),
            Qubit(id="q1", x_um=5.0, y_um=0.0, frequency_mhz=5100.0, movable=True),
        ]
        with pytest.raises(ValueError, match="Duplicate logical connection"):
            Project(
                id="p1",
                name="demo",
                chip_width_um=100.0,
                chip_height_um=100.0,
                constraints=_make_constraints(),
                qubits=qubits,
                connections=[
                    Connection(source_qubit_id="q0", target_qubit_id="q1", interaction_weight=1.0, gate_count=1),
                    Connection(source_qubit_id="q1", target_qubit_id="q0", interaction_weight=1.0, gate_count=1),
                ],
            )


class TestLQS:
    def test_lqs_bounds(self):
        qubits = _make_qubits([(0.0, 0.0), (10.0, 0.0)])
        lqs = compute_lqs(qubits, [])
        assert 0.0 <= lqs <= 100.0

    def test_lqs_empty_layout_returns_perfect_score(self):
        assert compute_lqs([], []) == 100.0
        assert compute_lqs([_make_qubits([(0.0, 0.0)])[0]], []) == 100.0

    def test_lqs_perfect_layout(self):
        qubits = _make_qubits([(0.0, 0.0), (50.0, 50.0)])
        qubits[1].frequency_mhz = 6000.0
        lqs = compute_lqs(qubits, [])
        assert lqs == pytest.approx(100.0, abs=0.1)

    def test_lqs_single_qubit(self):
        qubits = _make_qubits([(0.0, 0.0)])
        lqs = compute_lqs(qubits, [])
        assert lqs == 100.0

    def test_lqs_zero_qubits(self):
        lqs = compute_lqs([], [])
        assert lqs == 100.0


class TestExplainability:
    def test_explain_returns_dict(self):
        result = RiskResult(
            source_qubit_id="q0",
            target_qubit_id="q1",
            distance_um=8.2,
            frequency_delta_mhz=18.0,
            spatial_risk=0.8,
            spectral_risk=0.7,
            base_interaction_risk=0.6,
            interaction_weight=0.0,
            unintended_risk=0.6,
            routing_cost=0.0,
            objective_penalty=0.75,
            severity="HIGH",
        )
        explanation = explain_risk(result)
        assert "pair" in explanation
        assert "severity" in explanation
        assert "reasons" in explanation
        assert explanation["severity"] == "HIGH"

    def test_explain_intended_pair(self):
        result = RiskResult(
            source_qubit_id="q0",
            target_qubit_id="q1",
            distance_um=20.0,
            frequency_delta_mhz=30.0,
            spatial_risk=0.4,
            spectral_risk=0.5,
            base_interaction_risk=0.3,
            interaction_weight=1.0,
            unintended_risk=0.0,
            routing_cost=0.5,
            objective_penalty=0.35,
            severity="MEDIUM",
        )
        explanation = explain_risk(result)
        assert any("routing" in r.lower() for r in explanation["reasons"])

    def test_explain_low_risk(self):
        result = RiskResult(
            source_qubit_id="q0",
            target_qubit_id="q1",
            distance_um=50.0,
            frequency_delta_mhz=100.0,
            spatial_risk=0.1,
            spectral_risk=0.1,
            base_interaction_risk=0.05,
            interaction_weight=0.0,
            unintended_risk=0.05,
            routing_cost=0.0,
            objective_penalty=0.05,
            severity="LOW",
        )
        explanation = explain_risk(result)
        assert explanation["reasons"] == ["Low overall risk"]

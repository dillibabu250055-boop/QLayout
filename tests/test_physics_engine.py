import unittest
from typing import List
import numpy as np
from models.schema import Qubit, Connection, RiskResult
from core.physics_engine import (
    compute_euclidean_distance,
    compute_normalized_distance,
    compute_spatial_risk,
    compute_spectral_risk,
    compute_base_interaction_risk,
    compute_unintended_risk,
    compute_routing_cost,
    compute_objective_penalty,
    build_risk_results,
    _build_arrays,
)


def _make_qubits() -> List[Qubit]:
    return [
        Qubit(id="q0", x_um=0.0, y_um=0.0, frequency_mhz=5000.0, movable=True),
        Qubit(id="q1", x_um=10.0, y_um=10.0, frequency_mhz=5100.0, movable=True),
        Qubit(id="q2", x_um=5.0, y_um=5.0, frequency_mhz=5050.0, movable=False),
    ]


def _make_connections() -> List[Connection]:
    return [
        Connection(source_qubit_id="q0", target_qubit_id="q1", interaction_weight=1.0, gate_count=10),
        Connection(source_qubit_id="q0", target_qubit_id="q2", interaction_weight=0.0, gate_count=5),
    ]


class TestEuclideanDistance(unittest.TestCase):
    def test_symmetry(self):
        qubits = _make_qubits()
        conns = _make_connections()
        d = compute_euclidean_distance(qubits, conns)
        self.assertTrue(np.allclose(d, d.T))

    def test_diagonal_zero(self):
        qubits = _make_qubits()
        conns = _make_connections()
        d = compute_euclidean_distance(qubits, conns)
        self.assertTrue(np.all(np.diag(d) == 0.0))

    def test_known_values(self):
        q0 = Qubit(id="a", x_um=0.0, y_um=0.0, frequency_mhz=5000.0, movable=True)
        q1 = Qubit(id="b", x_um=3.0, y_um=4.0, frequency_mhz=5000.0, movable=True)
        conns = [Connection(source_qubit_id="a", target_qubit_id="b", interaction_weight=1.0, gate_count=1)]
        d = compute_euclidean_distance([q0, q1], conns)
        self.assertAlmostEqual(float(d[0, 1]), 5.0)
        self.assertAlmostEqual(float(d[1, 0]), 5.0)


class TestNormalizedDistance(unittest.TestCase):
    def test_unit_distance(self):
        d_ij = np.array([[0.0, 10.0], [10.0, 0.0]])
        d_norm = compute_normalized_distance(d_ij)
        self.assertAlmostEqual(float(d_norm[0, 1]), 1.0)

    def test_zero_distance(self):
        d_ij = np.array([[0.0, 0.0], [0.0, 0.0]])
        d_norm = compute_normalized_distance(d_ij)
        self.assertTrue(np.all(d_norm == 0.0))


class TestSpatialRisk(unittest.TestCase):
    def test_bounds(self):
        qubits = _make_qubits()
        conns = _make_connections()
        d_ij = compute_euclidean_distance(qubits, conns)
        d_norm = compute_normalized_distance(d_ij)
        S = compute_spatial_risk(d_norm)
        self.assertTrue(np.all(S >= 0.0))
        self.assertTrue(np.all(S <= 1.0))

    def test_zero_distance_gives_one(self):
        d_norm = np.array([[0.0, 0.0], [0.0, 0.0]])
        S = compute_spatial_risk(d_norm)
        self.assertAlmostEqual(float(S[0, 1]), 1.0)

    def test_large_distance_tends_to_zero(self):
        d_norm = np.array([[0.0, 1e6], [1e6, 0.0]])
        S = compute_spatial_risk(d_norm)
        self.assertAlmostEqual(float(S[0, 1]), 0.0, places=6)

    def test_symmetry(self):
        qubits = _make_qubits()
        conns = _make_connections()
        d_ij = compute_euclidean_distance(qubits, conns)
        d_norm = compute_normalized_distance(d_ij)
        S = compute_spatial_risk(d_norm)
        self.assertTrue(np.allclose(S, S.T))


class TestSpectralRisk(unittest.TestCase):
    def test_bounds(self):
        qubits = _make_qubits()
        conns = _make_connections()
        _, _, freqs, _ = _build_arrays(qubits, conns)
        F = compute_spectral_risk(freqs)
        self.assertTrue(np.all(F > 0.0))
        self.assertTrue(np.all(F <= 1.0))

    def test_zero_frequency_delta_gives_one(self):
        freqs = np.array([5000.0, 5000.0, 5050.0])
        F = compute_spectral_risk(freqs)
        self.assertAlmostEqual(float(F[0, 1]), 1.0)

    def test_large_frequency_delta_tends_to_zero(self):
        freqs = np.array([0.0, 100.0, 0.0])
        F = compute_spectral_risk(freqs)
        val = float(F[0, 1])
        self.assertGreater(val, 0.0)
        self.assertLess(val, 1e-3)

    def test_symmetry(self):
        qubits = _make_qubits()
        conns = _make_connections()
        _, _, freqs, _ = _build_arrays(qubits, conns)
        F = compute_spectral_risk(freqs)
        self.assertTrue(np.allclose(F, F.T))


class TestBaseInteractionRisk(unittest.TestCase):
    def test_bounds(self):
        qubits = _make_qubits()
        conns = _make_connections()
        d_ij = compute_euclidean_distance(qubits, conns)
        d_norm = compute_normalized_distance(d_ij)
        S = compute_spatial_risk(d_norm)
        _, _, freqs, _ = _build_arrays(qubits, conns)
        F = compute_spectral_risk(freqs)
        R = compute_base_interaction_risk(S, F)
        self.assertTrue(np.all(R >= 0.0))
        self.assertTrue(np.all(R <= 1.0))

    def test_symmetry(self):
        S = np.array([[0.5, 0.8], [0.8, 0.5]])
        F = np.array([[1.0, 0.9], [0.9, 1.0]])
        R = compute_base_interaction_risk(S, F)
        self.assertTrue(np.allclose(R, R.T))


class TestUnintendedRisk(unittest.TestCase):
    def test_intended_pair_zero_unintended(self):
        R = np.array([[0.5, 0.8], [0.8, 0.5]])
        I = np.array([[0.0, 1.0], [1.0, 0.0]])
        R_unintended = compute_unintended_risk(R, I)
        self.assertAlmostEqual(float(R_unintended[0, 1]), 0.0)

    def test_unintended_pair_matches_base(self):
        R = np.array([[0.5, 0.8], [0.8, 0.5]])
        I = np.array([[0.0, 0.0], [0.0, 0.0]])
        R_unintended = compute_unintended_risk(R, I)
        self.assertTrue(np.allclose(R_unintended, R))

    def test_bounds(self):
        R = np.array([[0.5, 0.8], [0.8, 0.5]])
        I = np.array([[0.0, 0.5], [0.5, 0.0]])
        R_unintended = compute_unintended_risk(R, I)
        self.assertTrue(np.all(R_unintended >= 0.0))
        self.assertTrue(np.all(R_unintended <= 1.0))


class TestRoutingCost(unittest.TestCase):
    def test_intended_pair_at_reference_distance(self):
        d_norm = np.array([[0.0, 1.0], [1.0, 0.0]])
        I = np.array([[0.0, 1.0], [1.0, 0.0]])
        C = compute_routing_cost(I, d_norm)
        expected = 1.0 * (1.0 ** 2 / (1.0 + 1.0 ** 2))
        self.assertAlmostEqual(float(C[0, 1]), expected)

    def test_zero_distance_routing_cost(self):
        d_norm = np.array([[0.0, 0.0], [0.0, 0.0]])
        I = np.array([[0.0, 1.0], [1.0, 0.0]])
        C = compute_routing_cost(I, d_norm)
        self.assertAlmostEqual(float(C[0, 1]), 0.0)

    def test_unintended_pair_zero_routing(self):
        d_norm = np.array([[0.0, 2.0], [2.0, 0.0]])
        I = np.array([[0.0, 0.0], [0.0, 0.0]])
        C = compute_routing_cost(I, d_norm)
        self.assertTrue(np.all(C == 0.0))

    def test_bounds(self):
        d_norm = np.array([[0.0, 1.0], [1.0, 0.0]])
        I = np.array([[0.0, 1.0], [1.0, 0.0]])
        C = compute_routing_cost(I, d_norm)
        self.assertTrue(np.all(C >= 0.0))
        self.assertTrue(np.all(C < 1.0))


class TestObjectivePenalty(unittest.TestCase):
    def test_bounds(self):
        R_unintended = np.array([[0.0, 0.8], [0.8, 0.0]])
        C_routing = np.array([[0.0, 0.5], [0.5, 0.0]])
        P = compute_objective_penalty(R_unintended, C_routing)
        self.assertTrue(np.all(P >= 0.0))
        self.assertTrue(np.all(P <= 1.0))

    def test_known_value(self):
        R_unintended = np.array([[0.0, 0.0], [0.0, 0.0]])
        C_routing = np.array([[0.0, 1.0], [1.0, 0.0]])
        P = compute_objective_penalty(R_unintended, C_routing)
        self.assertAlmostEqual(float(P[0, 1]), 0.3)


class TestBuildRiskResults(unittest.TestCase):
    def test_returns_list(self):
        qubits = _make_qubits()
        conns = _make_connections()
        results = build_risk_results(qubits, conns)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)

    def test_unconnected_close_pair_detection(self):
        qubits = [
            Qubit(id="q0", x_um=0.0, y_um=0.0, frequency_mhz=5000.0, movable=True),
            Qubit(id="q1", x_um=5.0, y_um=0.0, frequency_mhz=5005.0, movable=True),
        ]
        results = build_risk_results(qubits, [])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].source_qubit_id, "q0")
        self.assertEqual(results[0].target_qubit_id, "q1")
        self.assertEqual(results[0].interaction_weight, 0.0)
        self.assertGreater(results[0].unintended_risk, 0.0)

    def test_fields_populated(self):
        qubits = _make_qubits()
        conns = _make_connections()
        results = build_risk_results(qubits, conns)
        for r in results:
            self.assertIsInstance(r, RiskResult)
            self.assertIn(r.severity, ["HIGH", "MEDIUM", "LOW"])

    def test_bounds(self):
        qubits = _make_qubits()
        conns = _make_connections()
        results = build_risk_results(qubits, conns)
        for r in results:
            self.assertGreaterEqual(r.spatial_risk, 0.0)
            self.assertLessEqual(r.spatial_risk, 1.0)
            self.assertGreater(r.spectral_risk, 0.0)
            self.assertLessEqual(r.spectral_risk, 1.0)
            self.assertGreaterEqual(r.base_interaction_risk, 0.0)
            self.assertLessEqual(r.base_interaction_risk, 1.0)
            self.assertGreaterEqual(r.objective_penalty, 0.0)
            self.assertLessEqual(r.objective_penalty, 1.0)


class TestConnectionValidation(unittest.TestCase):
    def test_negative_weight_raises(self):
        with self.assertRaises(ValueError):
            Connection(source_qubit_id="q0", target_qubit_id="q1", interaction_weight=-0.1, gate_count=1)

    def test_above_one_weight_raises(self):
        with self.assertRaises(ValueError):
            Connection(source_qubit_id="q0", target_qubit_id="q1", interaction_weight=1.1, gate_count=1)

    def test_zero_weight_valid(self):
        connection = Connection(source_qubit_id="q0", target_qubit_id="q1", interaction_weight=0.0, gate_count=1)
        self.assertEqual(connection.interaction_weight, 0.0)

    def test_one_weight_valid(self):
        connection = Connection(source_qubit_id="q0", target_qubit_id="q1", interaction_weight=1.0, gate_count=1)
        self.assertEqual(connection.interaction_weight, 1.0)



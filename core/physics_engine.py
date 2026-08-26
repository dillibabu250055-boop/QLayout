import numpy as np
from itertools import combinations
from typing import List, Tuple
from models.schema import Qubit, Connection, RiskResult


D_REF: float = 10.0
SIGMA_F: float = 20.0
W_C: float = 0.70
W_R: float = 0.30


def _build_arrays(qubits: List[Qubit], connections: List[Connection]) -> Tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    id_to_idx = {q.id: i for i, q in enumerate(qubits)}
    coords = np.array([[q.x_um, q.y_um] for q in qubits], dtype=float).reshape(-1, 2)
    freqs = np.array([q.frequency_mhz for q in qubits], dtype=float)
    n = len(qubits)
    I = np.zeros((n, n), dtype=float)
    for conn in connections:
        if conn.source_qubit_id not in id_to_idx or conn.target_qubit_id not in id_to_idx:
            continue
        i = id_to_idx[conn.source_qubit_id]
        j = id_to_idx[conn.target_qubit_id]
        val = float(conn.interaction_weight)
        I[i, j] = val
        I[j, i] = val
    return id_to_idx, coords, freqs, I


def compute_euclidean_distance(qubits: List[Qubit], connections: List[Connection]) -> np.ndarray:
    if len(qubits) < 2:
        return np.zeros((len(qubits), len(qubits)), dtype=float)
    _, coords, _, _ = _build_arrays(qubits, connections)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    d_ij = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(d_ij, 0.0)
    return d_ij


def compute_normalized_distance(d_ij: np.ndarray) -> np.ndarray:
    return d_ij / D_REF


def compute_spatial_risk(d_norm: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + d_norm ** 3)


def compute_spectral_risk(freqs: np.ndarray) -> np.ndarray:
    delta_f = np.abs(freqs[:, np.newaxis] - freqs[np.newaxis, :])
    return np.exp(-delta_f ** 2 / (2.0 * SIGMA_F ** 2))


def compute_base_interaction_risk(spatial_risk: np.ndarray, spectral_risk: np.ndarray) -> np.ndarray:
    return spatial_risk * spectral_risk


def compute_unintended_risk(R: np.ndarray, I: np.ndarray) -> np.ndarray:
    return (1.0 - I) * R


def compute_routing_cost(I: np.ndarray, d_norm: np.ndarray) -> np.ndarray:
    return I * (d_norm ** 2 / (1.0 + d_norm ** 2))


def compute_objective_penalty(R_unintended: np.ndarray, C_routing: np.ndarray) -> np.ndarray:
    return W_C * R_unintended + W_R * C_routing


def build_risk_results(qubits: List[Qubit], connections: List[Connection]) -> List[RiskResult]:
    if len(qubits) < 2:
        return []

    id_to_idx, _, freqs, I = _build_arrays(qubits, connections)

    d_ij = compute_euclidean_distance(qubits, connections)
    d_norm = compute_normalized_distance(d_ij)
    S = compute_spatial_risk(d_norm)
    F = compute_spectral_risk(freqs)
    R = compute_base_interaction_risk(S, F)
    R_unintended = compute_unintended_risk(R, I)
    C_routing = compute_routing_cost(I, d_norm)
    P = compute_objective_penalty(R_unintended, C_routing)

    connection_map = {}
    for conn in connections:
        if conn.source_qubit_id not in id_to_idx or conn.target_qubit_id not in id_to_idx:
            continue
        i = id_to_idx[conn.source_qubit_id]
        j = id_to_idx[conn.target_qubit_id]
        if i == j:
            continue
        pair = tuple(sorted((i, j)))
        connection_map[pair] = float(conn.interaction_weight)

    from core.scoring import get_severity_from_penalty

    results: List[RiskResult] = []
    for i, j in combinations(range(len(qubits)), 2):
        pair = (i, j)
        interaction_weight = connection_map.get(pair, 0.0)
        penalty = float(P[i, j])
        sev = get_severity_from_penalty(penalty)

        results.append(RiskResult(
            source_qubit_id=qubits[i].id,
            target_qubit_id=qubits[j].id,
            distance_um=float(d_ij[i, j]),
            frequency_delta_mhz=float(np.abs(freqs[i] - freqs[j])),
            spatial_risk=float(S[i, j]),
            spectral_risk=float(F[i, j]),
            base_interaction_risk=float(R[i, j]),
            interaction_weight=interaction_weight,
            unintended_risk=float(R_unintended[i, j]),
            routing_cost=float(C_routing[i, j]),
            objective_penalty=penalty,
            severity=sev,
        ))
    return results

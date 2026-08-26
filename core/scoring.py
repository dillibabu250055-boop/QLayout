from typing import List, Dict, Any
import numpy as np

from models.schema import Qubit, Connection, RiskResult
from core.physics_engine import (
    _build_arrays,
    compute_euclidean_distance,
    compute_normalized_distance,
    compute_spatial_risk,
    compute_spectral_risk,
    compute_base_interaction_risk,
    compute_unintended_risk,
    compute_routing_cost,
    compute_objective_penalty,
)


def compute_lqs(qubits: List[Qubit], connections: List[Connection]) -> float:
    _, coords, freqs, I = _build_arrays(qubits, connections)
    n = len(qubits)
    if n < 2:
        return 100.0

    d_ij = compute_euclidean_distance(qubits, connections)
    d_norm = compute_normalized_distance(d_ij)
    S = compute_spatial_risk(d_norm)
    F = compute_spectral_risk(freqs)
    R = compute_base_interaction_risk(S, F)
    R_unintended = compute_unintended_risk(R, I)
    C_routing = compute_routing_cost(I, d_norm)
    P = compute_objective_penalty(R_unintended, C_routing)

    n_pairs = n * (n - 1) / 2.0
    if n_pairs == 0:
        return 100.0

    total_penalty = float(np.sum(np.triu(P, k=1))) / n_pairs
    lqs = 100.0 * (1.0 - total_penalty)
    return float(max(0.0, min(100.0, lqs)))


def explain_risk(result: RiskResult) -> Dict[str, Any]:
    explanation = {
        "pair": f"{result.source_qubit_id} - {result.target_qubit_id}",
        "severity": result.severity,
        "objective_penalty": round(result.objective_penalty, 4),
        "reasons": []
    }

    if result.spatial_risk > 0.7:
        explanation["reasons"].append(
            f"High spatial risk ({result.spatial_risk:.2f}): Qubits are only {result.distance_um:.1f}um apart"
        )
    elif result.spatial_risk > 0.3:
        explanation["reasons"].append(
            f"Moderate spatial risk ({result.spatial_risk:.2f}): Qubits are {result.distance_um:.1f}um apart"
        )

    if result.spectral_risk > 0.7:
        explanation["reasons"].append(
            f"High spectral risk ({result.spectral_risk:.2f}): Frequency separation is {result.frequency_delta_mhz:.1f}MHz"
        )
    elif result.spectral_risk > 0.3:
        explanation["reasons"].append(
            f"Moderate spectral risk ({result.spectral_risk:.2f}): Frequency separation is {result.frequency_delta_mhz:.1f}MHz"
        )

    if result.interaction_weight > 0.0:
        if result.routing_cost > 0.3:
            explanation["reasons"].append(
                f"Routing cost is high ({result.routing_cost:.2f}): Intended pair is far from reference distance"
            )
    else:
        if result.unintended_risk > 0.3:
            explanation["reasons"].append(
                f"Unintended coupling risk ({result.unintended_risk:.2f}): Non-intended pair has significant proximity"
            )

    if not explanation["reasons"]:
        explanation["reasons"].append("Low overall risk")

    return explanation

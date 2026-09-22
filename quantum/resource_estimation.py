"""Recompute pairwise leakage and quantum resource counts for the revised model."""

from __future__ import annotations

import json
import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from scheduling import campaign as mod


def pairwise_graph(
    instance: mod.Instance, mats: dict
) -> tuple[list[tuple[int, int]], dict[str, int]]:
    edges: list[tuple[int, int]] = []
    reasons = {"same_job": 0, "power": 0, "ramp": 0}
    n = len(instance.placements)
    headroom = mod.FEEDER_CAP_W - mod.N_NODES * mod.HARDWARE_IDLE_W_PER_NODE
    for i in range(n):
        pi = instance.placements[i]
        for j in range(i + 1, n):
            pj = instance.placements[j]
            reason = None
            if pi.job == pj.job:
                reason = "same_job"
            elif np.max(mats["power"][i] + mats["power"][j]) > headroom + 1e-9:
                reason = "power"
            else:
                for h, cap in mod.RAMP_CAPS_W_PER_S.items():
                    if (
                        np.max(mats["ramp_hi"][h][i] + mats["ramp_hi"][h][j])
                        > cap + 1e-9
                        or np.min(mats["ramp_lo"][h][i] + mats["ramp_lo"][h][j])
                        < -cap - 1e-9
                    ):
                        reason = "ramp"
                        break
            if reason is not None:
                edges.append((i, j))
                reasons[reason] += 1
    return edges, reasons


def solve_mwis(instance: mod.Instance, edges: list[tuple[int, int]]) -> dict:
    n = len(instance.placements)
    weights = np.array([p.weight for p in instance.placements], dtype=float)
    rows, cols, data = [], [], []
    for r, (i, j) in enumerate(edges):
        rows.extend([r, r])
        cols.extend([i, j])
        data.extend([1.0, 1.0])
    constraints = []
    if edges:
        A = coo_matrix((data, (rows, cols)), shape=(len(edges), n))
        constraints = [LinearConstraint(A, 0.0, 1.0)]
    result = milp(
        c=-weights,
        constraints=constraints,
        integrality=np.ones(n),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        options={"time_limit": 300.0, "presolve": True, "mip_rel_gap": 1e-9},
    )
    selected = (
        np.flatnonzero(result.x > 0.5).astype(int).tolist()
        if result.x is not None
        else []
    )
    return {
        "selected": selected,
        "work": float(weights[selected].sum()) if selected else 0.0,
        "certified": bool(
            result.status == 0 and result.mip_gap is not None and result.mip_gap <= 1e-8
        ),
        "mip_gap": None if result.mip_gap is None else float(result.mip_gap),
    }


def qubo_counts(
    instance: mod.Instance, mats_nominal: dict
) -> tuple[dict, np.ndarray, np.ndarray]:
    power = mats_nominal["power"]
    p_hat = power / mod.FEEDER_CAP_W
    gram_p = p_hat @ p_hat.T
    gram_r = np.zeros_like(gram_p)
    for h, cap in mod.RAMP_CAPS_W_PER_S.items():
        ramp = mats_nominal["ramp_hi"][h] / cap
        gram_r += ramp @ ramp.T
    gram_r /= len(mod.RAMP_CAPS_W_PER_S)
    coupling = 2.0 * (gram_p + gram_r)
    np.fill_diagonal(coupling, 0.0)
    iu = np.triu_indices(coupling.shape[0], 1)
    vals = coupling[iu]
    nz = vals[np.abs(vals) > 1e-12]
    adjacency = np.abs(coupling) > 1e-12
    degrees = adjacency.sum(axis=1)
    counts = {
        "logical_variables": int(coupling.shape[0]),
        "nonzero_couplings": int(nz.size),
        "negative_couplings": int(np.sum(nz < 0)),
        "positive_couplings": int(np.sum(nz > 0)),
        "coupling_density": float(nz.size / vals.size),
        "negative_fraction": float(np.mean(nz < 0)) if nz.size else 0.0,
        "ramp_only_negative_fraction": float(np.mean(gram_r[iu] < -1e-12)),
        "J_min": float(nz.min()) if nz.size else 0.0,
        "J_max": float(nz.max()) if nz.size else 0.0,
        "J_abs_median": float(np.median(np.abs(nz))) if nz.size else 0.0,
        "interaction_degree_min": int(degrees.min()) if degrees.size else 0,
        "interaction_degree_median": float(np.median(degrees)) if degrees.size else 0.0,
        "interaction_degree_mean": float(degrees.mean()) if degrees.size else 0.0,
        "interaction_degree_max": int(degrees.max()) if degrees.size else 0,
    }
    return counts, coupling, degrees


def main() -> None:
    cells, _, _ = mod.build_cells()
    instance = mod.build_instance(cells, 0)
    robust = mod.matrices(instance, cells, "robust")
    nominal = mod.matrices(instance, cells, "deterministic")
    edges, reasons = pairwise_graph(instance, robust)
    mwis = solve_mwis(instance, edges)
    mwis_metrics = mod.nominal_metrics(instance, robust, mwis["selected"])
    counts, coupling, degrees = qubo_counts(instance, nominal)

    delta_P = 500.0
    delta_R = 250.0
    headroom = mod.FEEDER_CAP_W - mod.N_NODES * mod.HARDWARE_IDLE_W_PER_NODE
    b_power = math.ceil(math.log2(headroom / delta_P + 1))
    b_ramp = {
        h: math.ceil(math.log2(2 * cap / delta_R + 1))
        for h, cap in mod.RAMP_CAPS_W_PER_S.items()
    }
    n_time = instance.horizon + 1
    slack_bits = n_time * (b_power + sum(b_ramp.values()))
    p = 3
    M = counts["nonzero_couplings"]
    Mneg = counts["negative_couplings"]
    delta_J = counts["interaction_degree_max"]
    resources = [
        {
            "architecture": "all-to-all signed",
            "logical_qubits": counts["logical_variables"],
            "qaoa_depth": p,
            "logical_zz_rotations": p * M,
            "cnot_equivalent_entanglers": 2 * p * M,
            "logical_two_qubit_depth_lower": p * delta_J,
            "logical_two_qubit_depth_upper": p * (delta_J + 1),
            "physical_qubit_lower_bound": counts["logical_variables"],
        },
        {
            "architecture": "sparse signed",
            "logical_qubits": counts["logical_variables"],
            "physical_qubit_lower_bound": counts["logical_variables"],
            "full_instance_compilation": None,
            "reduced_compilation_anchor": {
                "logical_variables": 8,
                "nonzero_couplings": 22,
                "qaoa_depth": 2,
                "all_to_all_entanglers": 44,
                "linear_nearest_neighbour_two_qubit_ops": 464,
                "swaps": 140,
                "ratio": 464 / 44,
            },
        },
        {
            "architecture": "positive unit-disk blockade",
            "logical_atoms": counts["logical_variables"],
            "negative_couplings": Mneg,
            "atom_sensitivity_eta_1": counts["logical_variables"] + Mneg,
            "constructive_embedding": False,
        },
    ]
    output = {
        "instance_seed": 0,
        "pairwise_graph": {
            "vertices": len(instance.placements),
            "edges": len(edges),
            "density": 2
            * len(edges)
            / (len(instance.placements) * (len(instance.placements) - 1)),
            "edge_reasons": reasons,
            "mwis_certified": mwis["certified"],
            "mwis_work": mwis["work"],
            "mwis_source_feasible": mwis_metrics["feasible"],
            "mwis_peak_over_cap": mwis_metrics["peak_over_cap"],
            "mwis_ramp_ratio_bound": mwis_metrics["ramp_ratio"],
        },
        "nominal_quadratic_proposal": counts,
        "exact_bounded_slack_accounting": {
            "power_resolution_W": delta_P,
            "ramp_resolution_W_per_s": delta_R,
            "power_bits_per_row": b_power,
            "ramp_bits_per_row": b_ramp,
            "sample_times": n_time,
            "power_slack_bits": n_time * b_power,
            "ramp_slack_bits_by_horizon": {
                str(h): n_time * bits for h, bits in b_ramp.items()
            },
            "uniform_slack_bits": slack_bits,
            "total_binary_variables": slack_bits + counts["logical_variables"],
        },
        "resource_estimates": resources,
        "interpretation": {
            "quadratic_model": "nominal observed-medoid proposal objective",
            "source_verifier": "robust total-IT power and multi-window ramp rows",
            "blockade_eta_1": "non-constructive sensitivity, not an embedding certificate",
        },
    }
    np.save(mod.RESULTS / "nominal_quadratic_coupling.npy", coupling)
    np.save(mod.RESULTS / "nominal_interaction_degrees.npy", degrees)
    (mod.RESULTS / "quantum_resource_revision.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

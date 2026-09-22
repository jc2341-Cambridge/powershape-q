"""Reviewer-driven revision experiments for the physical scheduling model.

The source NLR archive is read only.  Every derived artefact is written to the
repository-level ``results`` directory.  The experiment makes four deliberate changes to
the earlier campaign:

1. finite offline-inference batches replace online-serving traces as the
   schedulable work unit;
2. the 26 kW constraint is imposed on total IT power, including the measured
   dedicated-idle baseline reported by Vercellino et al.;
3. templates are observed medoid runs, so duration, energy, peak and shape
   originate from one run; and
4. schedules are replayed on held-out measured runs and compared with
   quantile-calibrated and robust training envelopes.

The identical node identities are eliminated from the binary variable index.
At most twelve intervals may overlap; an interval graph with clique number at
most twelve is twelve-colourable, so a node assignment can be recovered after
optimisation without changing the feasible set.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, hstack, vstack


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATASET = Path(
    os.environ.get("POWERSHAPE_Q_DATASET", str(ROOT / "data"))
).expanduser().resolve()
OFFLINE = DATASET / "01_aggregated_datasets" / "inference_offline_llama3_70b"

DT_S = 1.0
N_NODES = 12
NODES_PER_JOB = 1
N_GROUPS = N_NODES
FEEDER_CAP_W = 26_000.0
# Dedicated device-idle measurements: four GPUs at 72.5 W and two CPU sockets
# at 64.1 W.  The 418.2 W sum is kept rather than rounded to 420 W.
HARDWARE_IDLE_W_PER_NODE = 4 * 72.5 + 2 * 64.1
# One declared average ramp-rate limit is applied at three observation
# horizons.  This avoids manufacturing longer-window limits by rescaling a
# sample statistic from the same workload library.
RAMP_CAPS_W_PER_S = {1: 9_000.0, 5: 9_000.0, 30: 9_000.0}
TRANSITION_S = 2
BATCH_SIZES = (50, 100, 250, 500, 750, 1000)
OUTPUT_LENGTHS = (512, 1024)
MODES = ("deterministic", "quantile95", "robust")
# A 10% guard combines the published approximately 5% NVML accuracy band with
# a second 5% finite-sample envelope allowance.  It is fixed before campaign
# optimisation and applied only to the uncertainty-aware formulations.
METER_MARGIN = 0.10


@dataclass(frozen=True)
class RunTrace:
    run_id: int
    cell: tuple[int, int]
    power_W: np.ndarray
    energy_Wh: float
    duration_s: float


@dataclass
class CellModel:
    cell: tuple[int, int]
    train_ids: list[int]
    test_ids: list[int]
    medoid_id: int
    medoid_power_W: np.ndarray
    profile_W: dict[str, np.ndarray]
    ramp_bounds: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]
    occupancy_len: int
    service_ready_W: float


@dataclass(frozen=True)
class Placement:
    var_id: int
    job: int
    cell: tuple[int, int]
    start: int
    occupancy_len: int
    weight: float


@dataclass
class Instance:
    seed: int
    placements: list[Placement]
    job_cells: list[tuple[int, int]]
    release: np.ndarray
    deadline: np.ndarray
    horizon: int


def transition_complete(power: np.ndarray, idle_W: float, transition_s: int) -> np.ndarray:
    """Add explicit idle-to-active and active-to-idle linear transitions."""
    if power.size < 2:
        raise ValueError("trace must contain at least two samples")
    entry = np.linspace(idle_W, float(power[0]), transition_s + 1)[:-1]
    exit_ = np.linspace(float(power[-1]), idle_W, transition_s + 1)[1:]
    return np.concatenate([entry, power, exit_])


def read_run(run_id: int, meta: pd.Series, transition_s: int = TRANSITION_S) -> RunTrace:
    frame = pd.read_parquet(OFFLINE / "results" / f"{run_id:06d}.parquet")
    t = frame.index.to_numpy(dtype=float)
    p = frame["power[W]"].to_numpy(dtype=float)
    order = np.argsort(t)
    t, p = t[order], p[order]
    grid = np.arange(0.0, math.floor(float(t[-1])) + 1.0, DT_S)
    if grid.size < 2:
        grid = np.array([0.0, DT_S])
    sampled = np.interp(grid, t, p)
    complete = transition_complete(sampled, HARDWARE_IDLE_W_PER_NODE, transition_s)
    energy = float(np.trapz(complete, dx=DT_S) / 3600.0)
    return RunTrace(
        run_id=run_id,
        cell=(int(meta["batch_size"]), int(meta["max_output_tokens"])),
        power_W=complete,
        energy_Wh=energy,
        duration_s=float((complete.size - 1) * DT_S),
    )


def pad_incremental(traces: list[RunTrace], length: int) -> np.ndarray:
    arr = np.zeros((len(traces), length), dtype=float)
    for i, trace in enumerate(traces):
        q = np.maximum(trace.power_W - HARDWARE_IDLE_W_PER_NODE, 0.0)
        arr[i, : min(length, q.size)] = q[:length]
    return arr


def finite_difference_rows(values: np.ndarray, h: int) -> np.ndarray:
    lagged = np.zeros_like(values)
    if h < values.shape[-1]:
        lagged[..., h:] = values[..., :-h]
    return (values - lagged) / (h * DT_S)


def build_cells(
    split_seed: int = 20260921,
    transition_s: int = TRANSITION_S,
) -> tuple[dict[tuple[int, int], CellModel], dict[int, RunTrace], pd.DataFrame]:
    metadata_path = OFFLINE / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            "The NLR offline-inference dataset was not found at "
            f"{OFFLINE}. Set POWERSHAPE_Q_DATASET to the dataset root."
        )
    meta = pd.read_csv(metadata_path).reset_index(drop=True)
    selected = meta[
        meta["batch_size"].isin(BATCH_SIZES)
        & meta["max_output_tokens"].isin(OUTPUT_LENGTHS)
    ].copy()
    selected["run_id"] = selected.index.astype(int)

    traces: dict[int, RunTrace] = {}
    for _, row in selected.iterrows():
        rid = int(row["run_id"])
        traces[rid] = read_run(rid, row, transition_s=transition_s)

    rng = np.random.default_rng(split_seed)
    cells: dict[tuple[int, int], CellModel] = {}
    audit_rows: list[dict] = []
    for cell, group in selected.groupby(["batch_size", "max_output_tokens"], sort=True):
        cell = (int(cell[0]), int(cell[1]))
        ids = group["run_id"].astype(int).to_numpy()
        ids = ids[rng.permutation(ids.size)]
        n_test = max(3, int(round(ids.size / 3)))
        test_ids = ids[:n_test].tolist()
        train_ids = ids[n_test:].tolist()
        train = [traces[i] for i in train_ids]
        test = [traces[i] for i in test_ids]
        longest = max(t.power_W.size for t in train)
        train_q = pad_incremental(train, longest)

        # The medoid is an observed run nearest to the pointwise training median
        # on a common, idle-padded physical-time grid.
        median = np.median(train_q, axis=0)
        medoid_pos = int(np.argmin(np.mean((train_q - median[None, :]) ** 2, axis=1)))
        medoid = train[medoid_pos]

        profile_q = {
            "deterministic": pad_incremental([medoid], longest)[0],
            "quantile95": (1.0 + METER_MARGIN) * np.quantile(train_q, 0.95, axis=0),
            "robust": (1.0 + METER_MARGIN) * np.max(train_q, axis=0),
        }
        profile_W = {
            mode: HARDWARE_IDLE_W_PER_NODE + q for mode, q in profile_q.items()
        }
        ramp_bounds: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {
            m: {} for m in MODES
        }
        # Retain at least the longest ramp horizon after the final active
        # sample.  Without these trailing idle samples, a long-window
        # completion ramp is silently truncated at the template boundary.
        ramp_input = np.pad(train_q, ((0, 0), (0, max(RAMP_CAPS_W_PER_S))))
        medoid_input = np.pad(
            profile_q["deterministic"], (0, max(RAMP_CAPS_W_PER_S))
        )
        for h in RAMP_CAPS_W_PER_S:
            run_ramps = finite_difference_rows(ramp_input, h)
            medoid_ramp = finite_difference_rows(medoid_input, h)
            ramp_bounds["deterministic"][h] = (medoid_ramp, medoid_ramp)
            qlo = np.quantile(run_ramps, 0.05, axis=0)
            qhi = np.quantile(run_ramps, 0.95, axis=0)
            rlo = np.min(run_ramps, axis=0)
            rhi = np.max(run_ramps, axis=0)
            ramp_bounds["quantile95"][h] = (
                qlo - METER_MARGIN * np.abs(qlo),
                qhi + METER_MARGIN * np.abs(qhi),
            )
            ramp_bounds["robust"][h] = (
                rlo - METER_MARGIN * np.abs(rlo),
                rhi + METER_MARGIN * np.abs(rhi),
            )

        service_ready = float(np.median([t.power_W[transition_s] for t in train]))
        cells[cell] = CellModel(
            cell=cell,
            train_ids=train_ids,
            test_ids=test_ids,
            medoid_id=medoid.run_id,
            medoid_power_W=medoid.power_W,
            profile_W=profile_W,
            ramp_bounds=ramp_bounds,
            occupancy_len=int(math.ceil((1.0 + METER_MARGIN) * longest)),
            service_ready_W=service_ready,
        )

        medoid_peak = float(np.max(medoid.power_W))
        medoid_energy = medoid.energy_Wh
        medoid_duration = medoid.duration_s
        for trace in test:
            audit_rows.append(
                {
                    "batch_size": cell[0],
                    "output_length": cell[1],
                    "run_id": trace.run_id,
                    "split": "test",
                    "medoid_run_id": medoid.run_id,
                    "duration_error_pct": 100 * (medoid_duration / trace.duration_s - 1),
                    "energy_error_pct": 100 * (medoid_energy / trace.energy_Wh - 1),
                    "peak_error_pct": 100 * (medoid_peak / np.max(trace.power_W) - 1),
                    "test_duration_s": trace.duration_s,
                    "test_energy_Wh": trace.energy_Wh,
                    "test_peak_W": float(np.max(trace.power_W)),
                }
            )
    return cells, traces, pd.DataFrame(audit_rows)


def build_instance(cells: dict[tuple[int, int], CellModel], seed: int) -> Instance:
    rng = np.random.default_rng(seed)
    keys = sorted(cells)
    n_jobs = 15
    horizon = 360
    slack = 90
    start_step = 10
    job_cells = [keys[int(i)] for i in rng.integers(0, len(keys), size=n_jobs)]
    release = np.zeros(n_jobs, dtype=int)
    deadline = np.zeros(n_jobs, dtype=int)
    placements: list[Placement] = []
    var_id = 0
    for job, cell in enumerate(job_cells):
        duration = cells[cell].occupancy_len
        latest_release = max(0, horizon - duration - slack)
        release[job] = int(rng.integers(0, latest_release + 1)) if latest_release else 0
        deadline[job] = min(horizon, release[job] + duration + slack)
        last_start = deadline[job] - duration
        starts = list(range(int(release[job]), int(last_start) + 1, start_step))
        if not starts:
            starts = [int(release[job])]
        weight = float(cell[0])  # admitted offline requests
        for start in starts:
            placements.append(
                Placement(var_id, job, cell, start, duration, weight)
            )
            var_id += 1
    return Instance(seed, placements, job_cells, release, deadline, horizon)


def matrices(instance: Instance, cells: dict[tuple[int, int], CellModel], mode: str) -> dict:
    n = len(instance.placements)
    T = instance.horizon + 1
    power = np.zeros((n, T), dtype=float)
    occ = np.zeros((n, T), dtype=float)
    ramp_lo = {h: np.zeros((n, T), dtype=float) for h in RAMP_CAPS_W_PER_S}
    ramp_hi = {h: np.zeros((n, T), dtype=float) for h in RAMP_CAPS_W_PER_S}
    jobs = np.zeros((len(instance.job_cells), n), dtype=float)
    for p in instance.placements:
        jobs[p.job, p.var_id] = 1.0
        cell = cells[p.cell]
        q = np.maximum(cell.profile_W[mode] - HARDWARE_IDLE_W_PER_NODE, 0.0)
        end = min(T, p.start + q.size)
        power[p.var_id, p.start:end] = NODES_PER_JOB * q[: end - p.start]
        oend = min(T, p.start + p.occupancy_len)
        occ[p.var_id, p.start:oend] = 1.0
        for h in RAMP_CAPS_W_PER_S:
            lo, hi = cell.ramp_bounds[mode][h]
            rend = min(T, p.start + lo.size)
            ramp_lo[h][p.var_id, p.start:rend] = NODES_PER_JOB * lo[: rend - p.start]
            ramp_hi[h][p.var_id, p.start:rend] = NODES_PER_JOB * hi[: rend - p.start]
    return {"power": power, "occ": occ, "ramp_lo": ramp_lo, "ramp_hi": ramp_hi, "jobs": jobs}


def base_constraints(instance: Instance, mats: dict) -> list[LinearConstraint]:
    headroom = FEEDER_CAP_W - N_NODES * HARDWARE_IDLE_W_PER_NODE
    constraints = [
        LinearConstraint(csr_matrix(mats["jobs"]), 0.0, 1.0),
        LinearConstraint(csr_matrix(mats["occ"].T), -np.inf, float(N_GROUPS)),
        LinearConstraint(csr_matrix(mats["power"].T), -np.inf, headroom),
    ]
    for h, cap in RAMP_CAPS_W_PER_S.items():
        constraints.append(
            LinearConstraint(csr_matrix(mats["ramp_hi"][h].T), -np.inf, cap)
        )
        constraints.append(
            LinearConstraint(csr_matrix(mats["ramp_lo"][h].T), -cap, np.inf)
        )
    return constraints


def solve_work(instance: Instance, mats: dict, time_limit_s: float) -> dict:
    weights = np.array([p.weight for p in instance.placements], dtype=float)
    result = milp(
        c=-weights,
        constraints=base_constraints(instance, mats),
        integrality=np.ones(weights.size),
        bounds=Bounds(np.zeros(weights.size), np.ones(weights.size)),
        options={"time_limit": time_limit_s, "presolve": True, "mip_rel_gap": 1e-9},
    )
    selected = [] if result.x is None else np.flatnonzero(result.x > 0.5).astype(int).tolist()
    return {
        "selected": selected,
        "status": int(result.status),
        "message": str(result.message),
        "mip_gap": None if result.mip_gap is None else float(result.mip_gap),
        "certified": bool(result.status == 0 and result.mip_gap is not None and result.mip_gap <= 1e-8),
        "work": float(weights[selected].sum()) if selected else 0.0,
    }


def fixed_job_constraint(instance: Instance, mats: dict, selected: list[int]) -> LinearConstraint:
    jobs_selected = {instance.placements[v].job for v in selected}
    target = np.array(
        [1.0 if j in jobs_selected else 0.0 for j in range(len(instance.job_cells))]
    )
    return LinearConstraint(csr_matrix(mats["jobs"]), target, target)


def solve_peak(instance: Instance, mats: dict, selected: list[int], time_limit_s: float) -> dict:
    n = len(instance.placements)
    constraints: list[LinearConstraint] = []
    for con in base_constraints(instance, mats):
        constraints.append(
            LinearConstraint(hstack([con.A, csr_matrix((con.A.shape[0], 1))]), con.lb, con.ub)
        )
    fixed = fixed_job_constraint(instance, mats, selected)
    constraints.append(
        LinearConstraint(hstack([fixed.A, csr_matrix((fixed.A.shape[0], 1))]), fixed.lb, fixed.ub)
    )
    peak_rows = hstack(
        [csr_matrix(mats["power"].T), -np.ones((instance.horizon + 1, 1))],
        format="csr",
    )
    constraints.append(LinearConstraint(peak_rows, -np.inf, 0.0))
    c = np.zeros(n + 1)
    c[-1] = 1.0
    result = milp(
        c=c,
        constraints=constraints,
        integrality=np.r_[np.ones(n), 0.0],
        bounds=Bounds(np.zeros(n + 1), np.r_[np.ones(n), FEEDER_CAP_W]),
        options={"time_limit": time_limit_s, "presolve": True, "mip_rel_gap": 1e-9},
    )
    chosen = [] if result.x is None else np.flatnonzero(result.x[:n] > 0.5).astype(int).tolist()
    return {"selected": chosen, "status": int(result.status), "certified": bool(result.status == 0), "incremental_peak_W": None if result.x is None else float(result.x[-1])}


def solve_earliest(instance: Instance, mats: dict, selected: list[int], time_limit_s: float) -> dict:
    n = len(instance.placements)
    starts = np.array([p.start for p in instance.placements], dtype=float)
    constraints = base_constraints(instance, mats) + [fixed_job_constraint(instance, mats, selected)]
    result = milp(
        c=starts,
        constraints=constraints,
        integrality=np.ones(n),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        options={"time_limit": time_limit_s, "presolve": True, "mip_rel_gap": 1e-9},
    )
    chosen = [] if result.x is None else np.flatnonzero(result.x > 0.5).astype(int).tolist()
    return {"selected": chosen, "status": int(result.status), "certified": bool(result.status == 0)}


def nominal_metrics(instance: Instance, mats: dict, selected: list[int]) -> dict:
    idx = np.asarray(selected, dtype=int)
    baseline = N_NODES * HARDWARE_IDLE_W_PER_NODE
    if idx.size == 0:
        return {"work": 0.0, "peak_W": baseline, "ramp_ratio": 0.0, "feasible": True}
    load = mats["power"][idx].sum(axis=0)
    peak = baseline + float(np.max(load))
    # Deterministic matrices have identical lower/upper coefficients.  For the
    # uncertainty modes this reports the conservative bound used by the solver.
    ratios = []
    for h, cap in RAMP_CAPS_W_PER_S.items():
        upper = mats["ramp_hi"][h][idx].sum(axis=0)
        lower = mats["ramp_lo"][h][idx].sum(axis=0)
        ratios.append(max(float(np.max(upper)), float(np.max(-lower))) / cap)
    work = float(sum(instance.placements[v].weight for v in selected))
    occ = float(np.max(mats["occ"][idx].sum(axis=0)))
    return {
        "work": work,
        "peak_W": peak,
        "peak_over_cap": peak / FEEDER_CAP_W,
        "ramp_ratio": max(ratios),
        "max_concurrency": occ,
        "feasible": bool(peak <= FEEDER_CAP_W + 1e-6 and max(ratios) <= 1 + 1e-9 and occ <= N_GROUPS + 1e-9),
    }


def replay_once(instance: Instance, selected: list[int], cells: dict, traces: dict[int, RunTrace], rng: np.random.Generator) -> dict:
    if not selected:
        return {"peak_W": N_NODES * HARDWARE_IDLE_W_PER_NODE, "power_violation": False, "ramp_violation": False, "concurrency_violation": False, "deadline_violation": False, "feasible": True}
    max_len = instance.horizon + max(cells[instance.placements[v].cell].occupancy_len for v in selected) + 20
    q_total = np.zeros(max_len, dtype=float)
    occupancy = np.zeros(max_len, dtype=float)
    deadline_violation = False
    for v in selected:
        p = instance.placements[v]
        cell = cells[p.cell]
        rid = int(rng.choice(cell.test_ids))
        trace = traces[rid]
        q = NODES_PER_JOB * np.maximum(trace.power_W - HARDWARE_IDLE_W_PER_NODE, 0.0)
        end = p.start + q.size
        if end > q_total.size:
            extra = end - q_total.size
            q_total = np.pad(q_total, (0, extra))
            occupancy = np.pad(occupancy, (0, extra))
        q_total[p.start:end] += q
        occupancy[p.start:end] += 1.0
        deadline_violation |= bool(end - 1 > instance.deadline[p.job])
    total = N_NODES * HARDWARE_IDLE_W_PER_NODE + q_total
    power_violation = bool(np.max(total) > FEEDER_CAP_W + 1e-6)
    ramp_ratios = []
    for h, cap in RAMP_CAPS_W_PER_S.items():
        ramp = finite_difference_rows(q_total, h)
        ramp_ratios.append(float(np.max(np.abs(ramp))) / cap)
    ramp_violation = bool(max(ramp_ratios) > 1 + 1e-9)
    concurrency_violation = bool(np.max(occupancy) > N_GROUPS + 1e-9)
    return {
        "peak_W": float(np.max(total)),
        "peak_over_cap": float(np.max(total) / FEEDER_CAP_W),
        "max_ramp_ratio": max(ramp_ratios),
        "max_concurrency": float(np.max(occupancy)),
        "power_violation": power_violation,
        "ramp_violation": ramp_violation,
        "concurrency_violation": concurrency_violation,
        "deadline_violation": deadline_violation,
        "feasible": not (power_violation or ramp_violation or concurrency_violation or deadline_violation),
    }


def run_seed_experiments(
    seed: int,
    cells: dict[tuple[int, int], CellModel],
    traces: dict[int, RunTrace],
    replays: int,
    time_limit_s: float,
) -> tuple[list[dict], list[dict], dict[str, dict]]:
    rows: list[dict] = []
    replay_rows: list[dict] = []
    schedules: dict[str, dict] = {}
    instance = build_instance(cells, seed)
    for mode in MODES:
        mats = matrices(instance, cells, mode)
        t0 = time.perf_counter()
        work = solve_work(instance, mats, time_limit_s)
        peak = solve_peak(instance, mats, work["selected"], time_limit_s)
        earliest = solve_earliest(instance, mats, work["selected"], time_limit_s)
        elapsed = time.perf_counter() - t0
        peak_metrics = nominal_metrics(instance, mats, peak["selected"])
        earliest_metrics = nominal_metrics(instance, mats, earliest["selected"])
        reduction = 100.0 * (earliest_metrics["peak_W"] - peak_metrics["peak_W"]) / earliest_metrics["peak_W"] if earliest_metrics["peak_W"] else 0.0
        rows.append(
            {
                "seed": seed,
                "mode": mode,
                "n_variables": len(instance.placements),
                "work": work["work"],
                "n_admitted_jobs": len({instance.placements[v].job for v in work["selected"]}),
                "work_certified": work["certified"],
                "work_mip_gap": work["mip_gap"],
                "peak_certified": peak["certified"],
                "earliest_certified": earliest["certified"],
                "peak_W": peak_metrics["peak_W"],
                "earliest_feasible_peak_W": earliest_metrics["peak_W"],
                "paired_peak_reduction_pct": reduction,
                "ramp_ratio_bound": peak_metrics["ramp_ratio"],
                "solve_s": elapsed,
            }
        )
        schedules[f"{seed}:{mode}"] = {
            "job_cells": [list(c) for c in instance.job_cells],
            "release_s": instance.release.tolist(),
            "deadline_s": instance.deadline.tolist(),
            "selected": peak["selected"],
            "placements": [
                {"var_id": p.var_id, "job": p.job, "cell": list(p.cell), "start_s": p.start, "occupancy_len": p.occupancy_len, "weight": p.weight}
                for p in instance.placements
            ],
        }
        rng = np.random.default_rng(100_000 + 10_000 * seed + MODES.index(mode))
        for rep in range(replays):
            record = replay_once(instance, peak["selected"], cells, traces, rng)
            replay_rows.append({"seed": seed, "mode": mode, "replay": rep, **record})
    return rows, replay_rows, schedules


def run_campaign(n_seeds: int, replays: int, time_limit_s: float, workers: int) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    cells, traces, template_audit = build_cells()
    template_audit.to_csv(RESULTS / "heldout_template_errors.csv", index=False)
    rows: list[dict] = []
    replay_rows: list[dict] = []
    schedules: dict[str, dict] = {}
    if workers <= 1:
        for seed in range(n_seeds):
            r, rr, ss = run_seed_experiments(seed, cells, traces, replays, time_limit_s)
            rows.extend(r)
            replay_rows.extend(rr)
            schedules.update(ss)
            print(f"seed {seed:02d}/{n_seeds - 1:02d} complete", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(run_seed_experiments, seed, cells, traces, replays, time_limit_s): seed
                for seed in range(n_seeds)
            }
            for future in as_completed(futures):
                seed = futures[future]
                r, rr, ss = future.result()
                rows.extend(r)
                replay_rows.extend(rr)
                schedules.update(ss)
                print(f"seed {seed:02d}/{n_seeds - 1:02d} complete", flush=True)

    campaign = pd.DataFrame(rows)
    replay = pd.DataFrame(replay_rows)
    campaign.to_csv(RESULTS / "campaign.csv", index=False)
    replay.to_csv(RESULTS / "heldout_replay.csv", index=False)
    (RESULTS / "schedules.json").write_text(json.dumps(schedules, indent=2), encoding="utf-8")

    state_rows = []
    within_run_q02 = []
    first_active = []
    for trace in traces.values():
        core = trace.power_W[TRANSITION_S:-TRANSITION_S]
        within_run_q02.append(float(np.quantile(core, 0.02)))
        first_active.append(float(core[0]))
    for name, idle in (
        ("dedicated_hardware_idle", HARDWARE_IDLE_W_PER_NODE),
        ("service_ready_first_sample", float(np.median(first_active))),
        ("within_run_second_percentile", float(np.median(within_run_q02))),
        ("legacy_online_within_run_second_percentile", 1885.639894139787),
    ):
        state_rows.append(
            {
                "state": name,
                "per_node_W": idle,
                "twelve_node_baseline_kW": N_NODES * idle / 1000.0,
                "headroom_under_26kW_kW": (FEEDER_CAP_W - N_NODES * idle) / 1000.0,
            }
        )
    pd.DataFrame(state_rows).to_csv(RESULTS / "state_boundary_sensitivity.csv", index=False)

    replay_summary = (
        replay.groupby("mode")
        .agg(
            n=("feasible", "size"),
            feasible_rate=("feasible", "mean"),
            power_violation_rate=("power_violation", "mean"),
            ramp_violation_rate=("ramp_violation", "mean"),
            concurrency_violation_rate=("concurrency_violation", "mean"),
            deadline_violation_rate=("deadline_violation", "mean"),
            median_peak_over_cap=("peak_over_cap", "median"),
            p95_peak_over_cap=("peak_over_cap", lambda x: x.quantile(0.95)),
            median_max_ramp_ratio=("max_ramp_ratio", "median"),
            p95_max_ramp_ratio=("max_ramp_ratio", lambda x: x.quantile(0.95)),
        )
        .reset_index()
    )
    campaign_summary = (
        campaign.groupby("mode")
        .agg(
            n_instances=("seed", "size"),
            certification_rate=("work_certified", "mean"),
            peak_certification_rate=("peak_certified", "mean"),
            earliest_certification_rate=("earliest_certified", "mean"),
            median_work=("work", "median"),
            median_admitted_jobs=("n_admitted_jobs", "median"),
            median_peak_kW=("peak_W", lambda x: x.median() / 1000.0),
            median_earliest_peak_kW=("earliest_feasible_peak_W", lambda x: x.median() / 1000.0),
            median_paired_peak_reduction_pct=("paired_peak_reduction_pct", "median"),
            q25_paired_peak_reduction_pct=("paired_peak_reduction_pct", lambda x: x.quantile(0.25)),
            q75_paired_peak_reduction_pct=("paired_peak_reduction_pct", lambda x: x.quantile(0.75)),
            median_solve_s=("solve_s", "median"),
        )
        .reset_index()
    )
    summary = {
        "evidence": {
            "workload": "measured offline-inference batches",
            "selected_cells": [list(c) for c in sorted(cells)],
            "runs_per_cell": 15,
            "train_runs_per_cell": 10,
            "heldout_runs_per_cell": 5,
            "template": "observed training medoid",
            "transition": f"linear {TRANSITION_S}-s idle/active boundary appended to each measured run",
            "meter_margin": METER_MARGIN,
        },
        "physical_boundary": {
            "feeder_cap_W": FEEDER_CAP_W,
            "nodes": N_NODES,
            "dedicated_idle_W_per_node": HARDWARE_IDLE_W_PER_NODE,
            "idle_baseline_W": N_NODES * HARDWARE_IDLE_W_PER_NODE,
            "incremental_headroom_W": FEEDER_CAP_W - N_NODES * HARDWARE_IDLE_W_PER_NODE,
            "ramp_caps_W_per_s": RAMP_CAPS_W_PER_S,
        },
        "formulation": {
            "jobs": 15,
            "binary_variables_per_instance": int(campaign["n_variables"].median()),
            "group_model": "identical-node concurrency row; interval colouring recovered post hoc",
            "lexicographic_objective": "maximise admitted offline requests, then minimise total-IT peak for the same admitted jobs",
            "comparison": "earliest feasible placement of the same admitted jobs under identical constraints",
        },
        "campaign": campaign_summary.to_dict(orient="records"),
        "heldout_replay": replay_summary.to_dict(orient="records"),
        "template_error": {
            "median_abs_duration_error_pct": float(template_audit["duration_error_pct"].abs().median()),
            "median_abs_energy_error_pct": float(template_audit["energy_error_pct"].abs().median()),
            "median_abs_peak_error_pct": float(template_audit["peak_error_pct"].abs().median()),
            "p95_abs_duration_error_pct": float(template_audit["duration_error_pct"].abs().quantile(0.95)),
            "p95_abs_energy_error_pct": float(template_audit["energy_error_pct"].abs().quantile(0.95)),
            "p95_abs_peak_error_pct": float(template_audit["peak_error_pct"].abs().quantile(0.95)),
        },
    }
    (RESULTS / "revision_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    run_campaign(args.seeds, args.replays, args.time_limit, args.workers)


if __name__ == "__main__":
    main()

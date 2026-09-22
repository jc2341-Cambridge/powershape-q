"""Replay certified robust schedules under state and transition alternatives."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "revision_experiments", HERE / "run_revision_experiments.py"
)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def instance_from_payload(seed: int, payload: dict) -> mod.Instance:
    placements = [
        mod.Placement(
            int(p["var_id"]),
            int(p["job"]),
            tuple(p["cell"]),
            int(p["start_s"]),
            int(p["occupancy_len"]),
            float(p["weight"]),
        )
        for p in payload["placements"]
    ]
    return mod.Instance(
        seed=seed,
        placements=placements,
        job_cells=[tuple(c) for c in payload["job_cells"]],
        release=np.asarray(payload["release_s"], dtype=int),
        deadline=np.asarray(payload["deadline_s"], dtype=int),
        horizon=360,
    )


def replay_with_idle(
    instance: mod.Instance,
    selected: list[int],
    cells: dict,
    traces: dict[int, mod.RunTrace],
    rng: np.random.Generator,
    idle_W: float,
) -> dict:
    max_len = instance.horizon + 180
    q_total = np.zeros(max_len, dtype=float)
    occupancy = np.zeros(max_len, dtype=float)
    deadline_violation = False
    for v in selected:
        p = instance.placements[v]
        rid = int(rng.choice(cells[p.cell].test_ids))
        trace = traces[rid]
        q = mod.NODES_PER_JOB * np.maximum(trace.power_W - idle_W, 0.0)
        end = p.start + q.size
        if end > q_total.size:
            extra = end - q_total.size
            q_total = np.pad(q_total, (0, extra))
            occupancy = np.pad(occupancy, (0, extra))
        q_total[p.start:end] += q
        occupancy[p.start:end] += 1
        deadline_violation |= bool(end - 1 > instance.deadline[p.job])
    total = mod.N_NODES * idle_W + q_total
    power_violation = bool(np.max(total) > mod.FEEDER_CAP_W + 1e-6)
    ratios = [
        float(np.max(np.abs(mod.finite_difference_rows(q_total, h)))) / cap
        for h, cap in mod.RAMP_CAPS_W_PER_S.items()
    ]
    ramp_violation = bool(max(ratios) > 1 + 1e-9)
    concurrency_violation = bool(np.max(occupancy) > mod.N_GROUPS + 1e-9)
    return {
        "power_violation": power_violation,
        "ramp_violation": ramp_violation,
        "concurrency_violation": concurrency_violation,
        "deadline_violation": deadline_violation,
        "feasible": not (
            power_violation
            or ramp_violation
            or concurrency_violation
            or deadline_violation
        ),
        "peak_over_cap": float(np.max(total) / mod.FEEDER_CAP_W),
        "max_ramp_ratio": max(ratios),
    }


def summarise(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    return (
        frame.groupby(key)
        .agg(
            n=("feasible", "size"),
            feasible_rate=("feasible", "mean"),
            power_violation_rate=("power_violation", "mean"),
            ramp_violation_rate=("ramp_violation", "mean"),
            deadline_violation_rate=("deadline_violation", "mean"),
            median_peak_over_cap=("peak_over_cap", "median"),
            p95_peak_over_cap=("peak_over_cap", lambda x: x.quantile(0.95)),
            p95_max_ramp_ratio=("max_ramp_ratio", lambda x: x.quantile(0.95)),
        )
        .reset_index()
    )


def main() -> None:
    schedules = json.loads(
        (mod.RESULTS / "schedules.json").read_text(encoding="utf-8")
    )
    transition_rows: list[dict] = []
    for transition_s in (1, 2, 5):
        cells, traces, _ = mod.build_cells(transition_s=transition_s)
        for seed in range(24):
            payload = schedules[f"{seed}:robust"]
            instance = instance_from_payload(seed, payload)
            rng = np.random.default_rng(500_000 + 10_000 * transition_s + seed)
            for rep in range(200):
                record = mod.replay_once(
                    instance, payload["selected"], cells, traces, rng
                )
                transition_rows.append(
                    {"transition_s": transition_s, "seed": seed, "replay": rep, **record}
                )
    transition = pd.DataFrame(transition_rows)
    transition.to_csv(mod.RESULTS / "transition_replay.csv", index=False)
    transition_summary = summarise(transition, "transition_s")
    transition_summary.to_csv(
        mod.RESULTS / "transition_sensitivity.csv", index=False
    )

    cells, traces, _ = mod.build_cells(transition_s=2)
    service_ready = float(np.median([c.service_ready_W for c in cells.values()]))
    states = {
        "dedicated_hardware_idle": mod.HARDWARE_IDLE_W_PER_NODE,
        "service_ready_first_sample": service_ready,
        "legacy_online_within_run_q02": 1885.639894139787,
    }
    state_rows: list[dict] = []
    for state, idle_W in states.items():
        for seed in range(24):
            payload = schedules[f"{seed}:robust"]
            instance = instance_from_payload(seed, payload)
            rng = np.random.default_rng(800_000 + 10_000 * list(states).index(state) + seed)
            for rep in range(200):
                record = replay_with_idle(
                    instance,
                    payload["selected"],
                    cells,
                    traces,
                    rng,
                    idle_W,
                )
                state_rows.append(
                    {
                        "state": state,
                        "idle_W_per_node": idle_W,
                        "baseline_kW": mod.N_NODES * idle_W / 1000,
                        "seed": seed,
                        "replay": rep,
                        **record,
                    }
                )
    state = pd.DataFrame(state_rows)
    state.to_csv(mod.RESULTS / "idle_state_replay.csv", index=False)
    state_summary = summarise(state, "state")
    state_summary.insert(
        1,
        "idle_W_per_node",
        state_summary["state"].map(states),
    )
    state_summary.to_csv(
        mod.RESULTS / "idle_state_sensitivity.csv", index=False
    )
    print("Transition sensitivity")
    print(transition_summary.to_string(index=False))
    print("\nState sensitivity")
    print(state_summary.to_string(index=False))


if __name__ == "__main__":
    main()

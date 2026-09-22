"""Held-out replay of robust admission schedules at binding feeder limits."""

from __future__ import annotations

import importlib.util
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "revision_experiments_capacity_replay", HERE / "run_revision_experiments.py"
)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)

CAPS_KW = (10, 12, 14)


def run_cap(cap_kW: int) -> list[dict]:
    mod.FEEDER_CAP_W = float(cap_kW * 1000)
    cells, traces, _ = mod.build_cells()
    rows: list[dict] = []
    for seed in range(24):
        instance = mod.build_instance(cells, seed)
        mats = mod.matrices(instance, cells, "robust")
        work = mod.solve_work(instance, mats, time_limit_s=300.0)
        if not work["certified"]:
            raise RuntimeError(f"uncertified cap={cap_kW}, seed={seed}")
        rng = np.random.default_rng(810000 + 1000 * cap_kW + seed)
        for replay_id in range(200):
            outcome = mod.replay_once(instance, work["selected"], cells, traces, rng)
            outcome.update(
                {
                    "cap_kW": cap_kW,
                    "seed": seed,
                    "replay": replay_id,
                    "admitted_requests": work["work"],
                    "admitted_jobs": len(work["selected"]),
                }
            )
            rows.append(outcome)
    return rows


def main() -> None:
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(run_cap, cap): cap for cap in CAPS_KW}
        for future in as_completed(futures):
            cap = futures[future]
            rows.extend(future.result())
            print(f"completed {cap} kW", flush=True)
    frame = pd.DataFrame(rows).sort_values(["cap_kW", "seed", "replay"])
    frame.to_csv(mod.RESULTS / "capacity_heldout_replay.csv", index=False)
    summary = (
        frame.groupby("cap_kW")
        .agg(
            replays=("replay", "count"),
            feasible_rate=("feasible", "mean"),
            power_violation_rate=("power_violation", "mean"),
            ramp_violation_rate=("ramp_violation", "mean"),
            deadline_violation_rate=("deadline_violation", "mean"),
            p95_peak_over_cap=("peak_over_cap", lambda x: x.quantile(0.95)),
            p95_max_ramp_ratio=("max_ramp_ratio", lambda x: x.quantile(0.95)),
        )
        .reset_index()
    )
    summary.to_csv(mod.RESULTS / "capacity_heldout_replay_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

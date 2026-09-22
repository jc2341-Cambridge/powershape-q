"""Held-out replay of robust admission schedules at binding feeder limits."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from scheduling import campaign as mod

CAPS_KW = (10, 12, 14)
N_SEEDS = 24
N_REPLAYS = 200
SOLVER_TIME_LIMIT_S = 300.0


def run_cap(cap_kw: int) -> list[dict]:
    """Optimise and replay the robust admission schedule at one cap."""

    mod.FEEDER_CAP_W = float(cap_kw * 1000)
    cells, traces, _ = mod.build_cells()
    rows: list[dict] = []
    for seed in range(N_SEEDS):
        instance = mod.build_instance(cells, seed)
        mats = mod.matrices(instance, cells, "robust")
        admission = mod.solve_work(instance, mats, time_limit_s=SOLVER_TIME_LIMIT_S)
        if not admission["certified"]:
            raise RuntimeError(
                f"admission stage is uncertified for cap={cap_kw} kW, seed={seed}"
            )
        admitted_jobs = {instance.placements[v].job for v in admission["selected"]}
        rng = np.random.default_rng(810000 + 1000 * cap_kw + seed)
        for replay_id in range(N_REPLAYS):
            outcome = mod.replay_once(
                instance,
                admission["selected"],
                cells,
                traces,
                rng,
            )
            outcome.update(
                {
                    "cap_kW": cap_kw,
                    "seed": seed,
                    "replay": replay_id,
                    "admitted_requests": admission["work"],
                    "admitted_jobs": len(admitted_jobs),
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

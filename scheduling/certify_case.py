"""Rerun one campaign mode with an extended peak-optimisation time limit."""

from __future__ import annotations

import argparse
import json

import numpy as np

from scheduling import campaign as mod


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=mod.MODES, required=True)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--replays", type=int, default=200)
    args = parser.parse_args()

    cells, traces, _ = mod.build_cells()
    instance = mod.build_instance(cells, args.seed)
    mats = mod.matrices(instance, cells, args.mode)
    work = mod.solve_work(instance, mats, args.time_limit)
    peak = mod.solve_peak(instance, mats, work["selected"], args.time_limit)
    earliest = mod.solve_earliest(instance, mats, work["selected"], args.time_limit)
    peak_metrics = mod.nominal_metrics(instance, mats, peak["selected"])
    earliest_metrics = mod.nominal_metrics(instance, mats, earliest["selected"])
    reduction = (
        100.0
        * (earliest_metrics["peak_W"] - peak_metrics["peak_W"])
        / earliest_metrics["peak_W"]
    )
    row = {
        "seed": args.seed,
        "mode": args.mode,
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
    }
    rng = np.random.default_rng(
        100_000 + 10_000 * args.seed + mod.MODES.index(args.mode)
    )
    replay = [
        {
            "seed": args.seed,
            "mode": args.mode,
            "replay": rep,
            **mod.replay_once(instance, peak["selected"], cells, traces, rng),
        }
        for rep in range(args.replays)
    ]
    schedule = {
        "job_cells": [list(c) for c in instance.job_cells],
        "release_s": instance.release.tolist(),
        "deadline_s": instance.deadline.tolist(),
        "selected": peak["selected"],
        "placements": [
            {
                "var_id": p.var_id,
                "job": p.job,
                "cell": list(p.cell),
                "start_s": p.start,
                "occupancy_len": p.occupancy_len,
                "weight": p.weight,
            }
            for p in instance.placements
        ],
    }
    out = mod.RESULTS / f"rerun_seed{args.seed}_{args.mode}.json"
    out.write_text(
        json.dumps({"row": row, "replay": replay, "schedule": schedule}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()

"""Certified robust admission sensitivity to the total-IT feeder limit.

All data, splits, templates, transition assumptions, release/deadline windows
and ramp constraints are inherited unchanged from ``scheduling.campaign``.
Only the total-IT feeder limit is varied.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from scheduling import campaign as mod

CAPS_KW = (10, 12, 14, 16, 18, 20, 22, 26)
N_SEEDS = 24
SOLVER_TIME_LIMIT_S = 300.0


def run_cap(cap_kw: int) -> list[dict]:
    """Solve the certified robust admission problem at one feeder cap."""

    mod.FEEDER_CAP_W = float(cap_kw * 1000)
    cells, _, _ = mod.build_cells()
    rows: list[dict] = []
    for seed in range(N_SEEDS):
        instance = mod.build_instance(cells, seed)
        mats = mod.matrices(instance, cells, "robust")
        admission = mod.solve_work(instance, mats, time_limit_s=SOLVER_TIME_LIMIT_S)
        if not admission["certified"]:
            raise RuntimeError(
                f"admission stage is uncertified for cap={cap_kw} kW, seed={seed}"
            )
        metrics = mod.nominal_metrics(instance, mats, admission["selected"])
        admitted_jobs = {instance.placements[v].job for v in admission["selected"]}
        rows.append(
            {
                "cap_kW": cap_kw,
                "seed": seed,
                "certified": True,
                "mip_gap": admission["mip_gap"],
                "admitted_requests": admission["work"],
                "admitted_jobs": len(admitted_jobs),
                "robust_peak_kW": metrics["peak_W"] / 1000.0,
                "robust_ramp_ratio": metrics["ramp_ratio"],
            }
        )
    return rows


def main() -> None:
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run_cap, cap): cap for cap in CAPS_KW}
        for future in as_completed(futures):
            cap = futures[future]
            result = future.result()
            rows.extend(result)
            print(f"completed {cap} kW", flush=True)

    frame = pd.DataFrame(rows).sort_values(["cap_kW", "seed"])
    out = mod.RESULTS / "capacity_sensitivity.csv"
    frame.to_csv(out, index=False)
    summary = (
        frame.groupby("cap_kW")
        .agg(
            instances=("seed", "count"),
            certification_rate=("certified", "mean"),
            median_requests=("admitted_requests", "median"),
            q25_requests=("admitted_requests", lambda x: x.quantile(0.25)),
            q75_requests=("admitted_requests", lambda x: x.quantile(0.75)),
            median_jobs=("admitted_jobs", "median"),
            median_peak_kW=("robust_peak_kW", "median"),
            median_ramp_ratio=("robust_ramp_ratio", "median"),
        )
        .reset_index()
    )
    summary.to_csv(mod.RESULTS / "capacity_sensitivity_summary.csv", index=False)
    (mod.RESULTS / "capacity_sensitivity_summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

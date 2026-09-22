"""Evaluate schedules that reserve an uncertain 1--5 s idle/active transition.

The source dataset is read only.  This analysis constructs a single robust
electrical envelope over every integer transition duration on the 1 s model
grid, re-solves the 24 scheduling instances, and replays the resulting
schedules on held-out traces completed with each admissible duration.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace

import numpy as np
import pandas as pd

from scheduling import campaign as mod


TAUS = tuple(range(1, 6))


def _pad(values: np.ndarray, length: int) -> np.ndarray:
    out = np.zeros(length, dtype=float)
    out[: values.size] = values
    return out


def build_transition_uncertainty():
    """Return a robust cell library over tau in {1,...,5} and replay traces."""
    by_tau = {}
    traces_by_tau = {}
    for tau in TAUS:
        cells, traces, _ = mod.build_cells(transition_s=tau)
        by_tau[tau] = cells
        traces_by_tau[tau] = traces

    merged = {}
    for key in sorted(by_tau[TAUS[-1]]):
        variants = [by_tau[tau][key] for tau in TAUS]
        base = variants[-1]

        profile_len = max(v.profile_W["robust"].size for v in variants)
        profiles = []
        for variant in variants:
            q = np.maximum(
                variant.profile_W["robust"] - mod.HARDWARE_IDLE_W_PER_NODE,
                0.0,
            )
            profiles.append(_pad(q, profile_len))
        robust_q = np.max(np.vstack(profiles), axis=0)

        robust_ramps = {}
        for horizon in mod.RAMP_CAPS_W_PER_S:
            ramp_len = max(v.ramp_bounds["robust"][horizon][0].size for v in variants)
            lows = []
            highs = []
            for variant in variants:
                lo, hi = variant.ramp_bounds["robust"][horizon]
                lows.append(_pad(lo, ramp_len))
                highs.append(_pad(hi, ramp_len))
            robust_ramps[horizon] = (
                np.min(np.vstack(lows), axis=0),
                np.max(np.vstack(highs), axis=0),
            )

        merged[key] = replace(
            base,
            profile_W={"robust": mod.HARDWARE_IDLE_W_PER_NODE + robust_q},
            ramp_bounds={"robust": robust_ramps},
            occupancy_len=max(v.occupancy_len for v in variants),
        )
    return merged, traces_by_tau


def run_seed(seed, cells, traces_by_tau, replays, time_limit_s):
    instance = mod.build_instance(cells, seed)
    mats = mod.matrices(instance, cells, "robust")
    t0 = time.perf_counter()
    work = mod.solve_work(instance, mats, time_limit_s)
    peak = mod.solve_peak(instance, mats, work["selected"], time_limit_s)
    earliest = mod.solve_earliest(instance, mats, work["selected"], time_limit_s)
    elapsed = time.perf_counter() - t0
    peak_metrics = mod.nominal_metrics(instance, mats, peak["selected"])
    earliest_metrics = mod.nominal_metrics(instance, mats, earliest["selected"])
    reduction = (
        100.0
        * (earliest_metrics["peak_W"] - peak_metrics["peak_W"])
        / earliest_metrics["peak_W"]
    )
    campaign = {
        "seed": seed,
        "transition_min_s": min(TAUS),
        "transition_max_s": max(TAUS),
        "work": work["work"],
        "n_admitted_jobs": len({instance.placements[v].job for v in work["selected"]}),
        "work_certified": work["certified"],
        "peak_certified": peak["certified"],
        "earliest_certified": earliest["certified"],
        "peak_W": peak_metrics["peak_W"],
        "earliest_feasible_peak_W": earliest_metrics["peak_W"],
        "paired_peak_reduction_pct": reduction,
        "ramp_ratio_bound": peak_metrics["ramp_ratio"],
        "solve_s": elapsed,
    }

    replay_rows = []
    for tau in TAUS:
        rng = np.random.default_rng(300_000 + 10_000 * seed)
        for rep in range(replays):
            record = mod.replay_once(
                instance,
                peak["selected"],
                cells,
                traces_by_tau[tau],
                rng,
            )
            replay_rows.append(
                {"seed": seed, "transition_s": tau, "replay": rep, **record}
            )
    return campaign, replay_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed-list", type=str, default="")
    parser.add_argument("--merge-existing", action="store_true")
    args = parser.parse_args()

    seeds = (
        [int(value) for value in args.seed_list.split(",") if value.strip()]
        if args.seed_list
        else list(range(args.seeds))
    )

    mod.RESULTS.mkdir(parents=True, exist_ok=True)
    cells, traces_by_tau = build_transition_uncertainty()
    campaigns = []
    replays = []
    if args.workers <= 1:
        for seed in seeds:
            campaign, replay = run_seed(
                seed, cells, traces_by_tau, args.replays, args.time_limit
            )
            campaigns.append(campaign)
            replays.extend(replay)
            print(f"seed {seed:02d}/{args.seeds - 1:02d} complete", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    run_seed,
                    seed,
                    cells,
                    traces_by_tau,
                    args.replays,
                    args.time_limit,
                ): seed
                for seed in seeds
            }
            for future in as_completed(futures):
                seed = futures[future]
                campaign, replay = future.result()
                campaigns.append(campaign)
                replays.extend(replay)
                print(f"seed {seed:02d}/{args.seeds - 1:02d} complete", flush=True)

    campaign_df = pd.DataFrame(campaigns)
    replay_df = pd.DataFrame(replays)
    campaign_path = mod.RESULTS / "transition_uncertainty_campaign.csv"
    replay_path = mod.RESULTS / "transition_uncertainty_replay.csv"
    if args.merge_existing and campaign_path.exists() and replay_path.exists():
        old_campaign = pd.read_csv(campaign_path)
        old_replay = pd.read_csv(replay_path)
        old_campaign = old_campaign[~old_campaign.seed.isin(seeds)]
        old_replay = old_replay[~old_replay.seed.isin(seeds)]
        campaign_df = pd.concat([old_campaign, campaign_df], ignore_index=True)
        replay_df = pd.concat([old_replay, replay_df], ignore_index=True)
    campaign_df = campaign_df.sort_values("seed")
    replay_df = replay_df.sort_values(["transition_s", "seed", "replay"])
    campaign_df.to_csv(campaign_path, index=False)
    replay_df.to_csv(replay_path, index=False)

    summary = (
        replay_df.groupby("transition_s")
        .agg(
            n=("feasible", "size"),
            feasible_rate=("feasible", "mean"),
            power_violation_rate=("power_violation", "mean"),
            ramp_violation_rate=("ramp_violation", "mean"),
            concurrency_violation_rate=("concurrency_violation", "mean"),
            deadline_violation_rate=("deadline_violation", "mean"),
            median_peak_over_cap=("peak_over_cap", "median"),
            p95_peak_over_cap=("peak_over_cap", lambda x: x.quantile(0.95)),
            p95_max_ramp_ratio=("max_ramp_ratio", lambda x: x.quantile(0.95)),
        )
        .reset_index()
    )
    summary.to_csv(mod.RESULTS / "transition_uncertainty_summary.csv", index=False)

    audit = {
        "transition_uncertainty_s": list(TAUS),
        "uncertainty_construction": (
            "pointwise upper/lower training envelope over all integer durations"
        ),
        "instances": int(campaign_df.seed.nunique()),
        "replays_per_duration": int(
            replay_df[replay_df.transition_s == min(TAUS)].shape[0]
        ),
        "all_stages_certified": bool(
            campaign_df[["work_certified", "peak_certified", "earliest_certified"]]
            .to_numpy()
            .all()
        ),
        "median_admitted_requests": float(campaign_df.work.median()),
        "median_admitted_jobs": float(campaign_df.n_admitted_jobs.median()),
        "median_paired_peak_reduction_pct": float(
            campaign_df.paired_peak_reduction_pct.median()
        ),
        "summary": summary.to_dict(orient="records"),
    }
    (mod.RESULTS / "transition_uncertainty_summary.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

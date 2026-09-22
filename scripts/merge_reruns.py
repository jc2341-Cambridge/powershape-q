"""Merge extended certified reruns and refresh summary tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    campaign = pd.read_csv(RESULTS / "campaign.csv")
    replay = pd.read_csv(RESULTS / "heldout_replay.csv")
    schedules = json.loads((RESULTS / "schedules.json").read_text(encoding="utf-8"))
    for path in sorted(RESULTS.glob("rerun_seed*_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = payload["row"]
        mask = (campaign["seed"] == row["seed"]) & (campaign["mode"] == row["mode"])
        if mask.sum() != 1:
            raise RuntimeError(f"cannot identify unique row for {path.name}")
        for key, value in row.items():
            if key in campaign.columns:
                campaign.loc[mask, key] = value
        replay = replay[~((replay["seed"] == row["seed"]) & (replay["mode"] == row["mode"]))]
        replay = pd.concat([replay, pd.DataFrame(payload["replay"])], ignore_index=True)
        schedules[f"{row['seed']}:{row['mode']}"] = payload["schedule"]

    campaign = campaign.sort_values(["seed", "mode"]).reset_index(drop=True)
    replay = replay.sort_values(["seed", "mode", "replay"]).reset_index(drop=True)
    campaign.to_csv(RESULTS / "campaign.csv", index=False)
    replay.to_csv(RESULTS / "heldout_replay.csv", index=False)
    (RESULTS / "schedules.json").write_text(json.dumps(schedules, indent=2), encoding="utf-8")

    summary = json.loads((RESULTS / "revision_summary.json").read_text(encoding="utf-8"))
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
    summary["campaign"] = campaign_summary.to_dict(orient="records")
    summary["heldout_replay"] = replay_summary.to_dict(orient="records")
    (RESULTS / "revision_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"campaign": summary["campaign"], "heldout_replay": summary["heldout_replay"]}, indent=2))


if __name__ == "__main__":
    main()

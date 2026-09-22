"""Compute the declared held-out deployment gate for feeder power risk."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CONFIDENCE = 0.95
MAX_POWER_VIOLATION_PROBABILITY = 0.01


def upper_clopper_pearson(k: int, n: int, confidence: float) -> float:
    if k >= n:
        return 1.0
    return float(beta.ppf(confidence, k + 1, n - k))


def main() -> None:
    tight = pd.read_csv(RESULTS / "capacity_heldout_replay.csv")
    primary = pd.read_csv(RESULTS / "heldout_replay.csv")
    rows = []
    for cap, frame in tight.groupby("cap_kW"):
        if cap not in (10, 12, 14):
            continue
        rows.append((float(cap), frame.power_violation.astype(bool)))
    rows.append(
        (
            26.0,
            primary[primary["mode"] == "robust"].power_violation.astype(bool),
        )
    )

    output = []
    for cap, violations in rows:
        n = int(violations.size)
        k = int(violations.sum())
        observed = k / n
        upper = upper_clopper_pearson(k, n, CONFIDENCE)
        output.append(
            {
                "cap_kW": cap,
                "replays": n,
                "power_violations": k,
                "observed_probability": observed,
                "upper_95_probability": upper,
                "risk_target": MAX_POWER_VIOLATION_PROBABILITY,
                "deployment_gate_pass": upper
                <= MAX_POWER_VIOLATION_PROBABILITY,
            }
        )
    result = pd.DataFrame(output).sort_values("cap_kW")
    result.to_csv(RESULTS / "capacity_risk_gate.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

"""Collect completed Braket tasks from a PowerShape-Q submission manifest."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from braket.aws import AwsQuantumTask

from .aquila_ahs import _decode_measurements
from .gate_model_qpu import _analyse_counts
from .problem_instance import REDUCED_INSTANCE
from .qaoa_circuit import canonical_counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect QPU results recorded in a task manifest."
    )
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not payload.get("submitted") or not payload.get("tasks"):
        raise SystemExit("manifest contains no submitted tasks")

    is_aquila = payload.get("target") == "aquila" or "AHS" in payload.get(
        "protocol", ""
    )
    for record in payload["tasks"]:
        task = AwsQuantumTask(record["task_arn"])
        result = task.result()
        record["state"] = task.state()
        if is_aquila:
            record.update(_decode_measurements(result.measurements))
        else:
            counts = canonical_counts(
                dict(result.measurement_counts), REDUCED_INSTANCE.n_variables
            )
            record["measurement_counts"] = counts
            record["top_candidates"] = _analyse_counts(counts)

    payload["collected_utc"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Updated {args.manifest}")


if __name__ == "__main__":
    main()

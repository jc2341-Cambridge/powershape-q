"""Shared Amazon Braket submission path for the gate-model QPU protocols."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from braket.aws import AwsDevice

from .problem_instance import REDUCED_INSTANCE
from .qaoa_circuit import (
    build_qaoa_circuit,
    canonical_counts,
    circuit_metadata,
    load_parameters,
)


DEFAULT_SHOTS = 4096
DEFAULT_REPETITIONS = 5


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _analyse_counts(counts: dict[str, int], top_k: int = 16) -> list[dict]:
    total = sum(counts.values())
    rows = []
    for bitstring, count in list(counts.items())[:top_k]:
        rows.append(
            {
                "bitstring": bitstring,
                "count": count,
                "probability": count / total,
                "qubo_energy": REDUCED_INSTANCE.energy(bitstring),
                "selected_weight": REDUCED_INSTANCE.selected_weight(bitstring),
                "job_unique": REDUCED_INSTANCE.job_unique(bitstring),
            }
        )
    return rows


def _parser(target: str) -> argparse.ArgumentParser:
    provider = "IONQ" if target == "ionq" else "RIGETTI"
    parser = argparse.ArgumentParser(
        description=f"Validate or submit the PowerShape-Q benchmark to {provider} hardware."
    )
    parser.add_argument("--depth", type=int, choices=(1, 2), default=2)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument(
        "--device-arn",
        default=os.getenv(f"POWERSHAPE_Q_{provider}_ARN"),
        help=f"Amazon Braket device ARN or POWERSHAPE_Q_{provider}_ARN.",
    )
    parser.add_argument("--s3-bucket", default=os.getenv("AMZN_BRAKET_TASK_RESULTS_S3_BUCKET"))
    parser.add_argument("--s3-prefix", default=f"powershape-q/{target}")
    parser.add_argument("--reservation-arn", default=None)
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Create paid QPU tasks. Without this flag the command only validates the protocol.",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for submitted tasks and store decoded measurement counts.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(target: str) -> None:
    args = _parser(target).parse_args()
    if args.shots <= 0 or args.repetitions <= 0:
        raise SystemExit("shots and repetitions must be positive")
    if args.wait and not args.submit:
        raise SystemExit("--wait requires --submit")
    if args.submit and not args.device_arn:
        raise SystemExit("--submit requires --device-arn or the matching environment variable")
    if args.device_arn and f"/{target}/" not in args.device_arn.lower():
        raise SystemExit(f"device ARN does not identify an {target} target")

    parameters = load_parameters(args.depth)
    synthesis = "zz" if target == "ionq" else "cnot"
    circuit = build_qaoa_circuit(parameters, two_qubit_synthesis=synthesis)
    stamp = _utc_stamp()
    output = args.output or Path("results") / "qpu" / f"{target}_p{args.depth}_{stamp}.json"
    payload = {
        "project": "PowerShape-Q",
        "protocol": "reduced signed QAOA",
        "target": target,
        "created_utc": stamp,
        "submitted": bool(args.submit),
        "device_arn": args.device_arn,
        "shots_per_task": args.shots,
        "independent_tasks": args.repetitions,
        "circuit": circuit_metadata(parameters, two_qubit_synthesis=synthesis),
        "tasks": [],
    }

    if not args.submit:
        payload["validation"] = {
            "qubo_to_ising": "passed",
            "circuit_instructions": len(circuit.instructions),
            "submission_flag_required": True,
        }
        _write_json(output, payload)
        print(json.dumps(payload, indent=2))
        print(f"Validation manifest written to {output}")
        return

    device = AwsDevice(args.device_arn)
    destination = (args.s3_bucket, args.s3_prefix) if args.s3_bucket else None
    for repetition in range(args.repetitions):
        task = device.run(
            circuit,
            s3_destination_folder=destination,
            shots=args.shots,
            reservation_arn=args.reservation_arn,
        )
        record = {
            "repetition": repetition,
            "task_arn": task.id,
            "state_at_submission": task.state(),
        }
        if args.wait:
            result = task.result()
            counts = canonical_counts(dict(result.measurement_counts), REDUCED_INSTANCE.n_variables)
            record.update(
                {
                    "state": task.state(),
                    "measurement_counts": counts,
                    "top_candidates": _analyse_counts(counts),
                }
            )
        payload["tasks"].append(record)
        _write_json(output, payload)
        print(f"Submitted {target} task {repetition + 1}/{args.repetitions}: {task.id}")

    print(f"Task manifest written to {output}")

"""QuEra Aquila AHS workflow for the reduced positive unit-disk graph."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
from braket.ahs.atom_arrangement import AtomArrangement
from braket.ahs.driving_field import DrivingField
from braket.ahs.field import Field
from braket.ahs.hamiltonian import Hamiltonian
from braket.ahs.local_detuning import LocalDetuning
from braket.ahs.pattern import Pattern
from braket.aws import AwsDevice
from braket.timings.time_series import TimeSeries

from .problem_instance import REDUCED_INSTANCE


DEFAULT_SHOTS = 1000
DEFAULT_REPETITIONS = 5
PRIMARY_EVOLUTION_US = 4.0
EVOLUTION_SENSITIVITIES_US = (2.0, 8.0)
OMEGA_MAX_RAD_S = 15.0e6
DETUNING_MAX_RAD_S = 15.0e6
C6_RAD_S_M6 = 5.42e-24
BLOCKADE_RADIUS_M = (C6_RAD_S_M6 / OMEGA_MAX_RAD_S) ** (1.0 / 6.0)

# Coordinates are in metres.  They realise the ten declared conflict edges
# inside the nominal blockade radius while retaining at least 4 micrometres
# between every pair of sites.
REGISTER_M = 1.16 * np.asarray(
    [
        [-14.919547119096561e-6, 6.135926368433351e-6],
        [11.85239843794084e-6, -4.083231271587238e-6],
        [1.2749048811140582e-6, 0.08096878032675216e-6],
        [-1.8232844416402525e-6, -5.266681521875945e-6],
        [-4.899222617965452e-6, 0.4972378464795923e-6],
        [-4.404941081482837e-6, -2.9413869587523302e-6],
        [5.25114284990158e-6, 5.038362526168229e-6],
        [7.668549091228623e-6, 0.5388042308075882e-6],
    ],
    dtype=float,
)

CONFLICT_EDGES = {
    (1, 7),
    (2, 3),
    (2, 4),
    (2, 5),
    (2, 6),
    (2, 7),
    (3, 4),
    (3, 5),
    (4, 5),
    (6, 7),
}


def validate_geometry() -> dict:
    realised: set[tuple[int, int]] = set()
    distances = []
    for i in range(len(REGISTER_M)):
        for j in range(i + 1, len(REGISTER_M)):
            distance = float(np.linalg.norm(REGISTER_M[i] - REGISTER_M[j]))
            distances.append(distance)
            if distance < BLOCKADE_RADIUS_M:
                realised.add((i, j))
    if realised != CONFLICT_EDGES:
        raise AssertionError("register geometry does not realise the declared graph")
    if min(distances) < 4.0e-6:
        raise AssertionError("register contains sites closer than 4 micrometres")
    return {
        "atoms": len(REGISTER_M),
        "conflict_edges": len(CONFLICT_EDGES),
        "blockade_radius_m": BLOCKADE_RADIUS_M,
        "minimum_site_separation_m": min(distances),
        "maximum_coordinate_m": float(np.max(np.abs(REGISTER_M))),
    }


def build_program(evolution_us: float) -> AnalogHamiltonianSimulation:
    if evolution_us not in (2.0, 4.0, 8.0):
        raise ValueError("evolution time must be 2, 4 or 8 microseconds")
    validate_geometry()

    duration = evolution_us * 1e-6
    times = (0.0, 0.1 * duration, 0.9 * duration, duration)
    amplitude = TimeSeries()
    detuning = TimeSeries()
    for time_s, value in zip(times, (0.0, OMEGA_MAX_RAD_S, OMEGA_MAX_RAD_S, 0.0)):
        amplitude.put(time_s, value)
    for time_s, value in zip(
        times,
        (-DETUNING_MAX_RAD_S, -DETUNING_MAX_RAD_S, DETUNING_MAX_RAD_S, DETUNING_MAX_RAD_S),
    ):
        detuning.put(time_s, value)
    phase = TimeSeries().put(0.0, 0.0).put(duration, 0.0)

    weights = np.asarray(REDUCED_INSTANCE.weights, dtype=float)
    span = float(np.ptp(weights))
    pattern = np.ones_like(weights) if span == 0 else 0.35 + 0.65 * (weights - weights.min()) / span
    local_magnitude = TimeSeries()
    for time_s, value in zip(times, (0.0, 0.0, 0.4 * DETUNING_MAX_RAD_S, 0.4 * DETUNING_MAX_RAD_S)):
        local_magnitude.put(time_s, value)

    register = AtomArrangement()
    for x, y in REGISTER_M:
        register.add([float(x), float(y)])

    hamiltonian = Hamiltonian()
    hamiltonian += DrivingField(amplitude=amplitude, phase=phase, detuning=detuning)
    hamiltonian += LocalDetuning(Field(local_magnitude, Pattern(pattern.tolist())))
    return AnalogHamiltonianSimulation(register=register, hamiltonian=hamiltonian)


def _independent(bitstring: str) -> bool:
    return all(not (bitstring[i] == "1" and bitstring[j] == "1") for i, j in CONFLICT_EDGES)


def _decode_measurements(measurements) -> dict:
    counts: Counter[str] = Counter()
    rejected = 0
    for measurement in measurements:
        pre = list(measurement.pre_sequence)
        post = list(measurement.post_sequence)
        if len(pre) != len(REGISTER_M) or len(post) != len(REGISTER_M) or not all(pre):
            rejected += 1
            continue
        bitstring = "".join(str(1 - int(value)) for value in post)
        counts[bitstring] += 1
    valid = sum(counts.values())
    rows = []
    for bitstring, count in counts.most_common(16):
        rows.append(
            {
                "bitstring": bitstring,
                "count": count,
                "probability_among_valid_shots": count / valid if valid else 0.0,
                "independent_set": _independent(bitstring),
                "selected_weight": REDUCED_INSTANCE.selected_weight(bitstring),
                "job_unique": REDUCED_INSTANCE.job_unique(bitstring),
            }
        )
    return {"valid_shots": valid, "rejected_shots": rejected, "top_candidates": rows}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate or submit the PowerShape-Q Aquila protocol."
    )
    parser.add_argument("--evolution-us", type=float, choices=(2.0, 4.0, 8.0), default=4.0)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--device-arn", default=os.getenv("POWERSHAPE_Q_AQUILA_ARN"))
    parser.add_argument("--s3-bucket", default=os.getenv("AMZN_BRAKET_TASK_RESULTS_S3_BUCKET"))
    parser.add_argument("--s3-prefix", default="powershape-q/aquila")
    parser.add_argument("--reservation-arn", default=None)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.shots <= 0 or args.repetitions <= 0:
        raise SystemExit("shots and repetitions must be positive")
    if args.wait and not args.submit:
        raise SystemExit("--wait requires --submit")
    if args.submit and not args.device_arn:
        raise SystemExit("--submit requires --device-arn or POWERSHAPE_Q_AQUILA_ARN")
    if args.device_arn and "/quera/" not in args.device_arn.lower():
        raise SystemExit("device ARN does not identify a QuEra target")

    programme = build_program(args.evolution_us)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("results") / "qpu" / f"aquila_{args.evolution_us:g}us_{stamp}.json"
    payload = {
        "project": "PowerShape-Q",
        "protocol": "reduced positive unit-disk AHS",
        "target": "aquila",
        "created_utc": stamp,
        "submitted": bool(args.submit),
        "device_arn": args.device_arn,
        "evolution_us": args.evolution_us,
        "shots_per_task": args.shots,
        "independent_tasks": args.repetitions,
        "omega_max_rad_s": OMEGA_MAX_RAD_S,
        "detuning_max_rad_s": DETUNING_MAX_RAD_S,
        "geometry": validate_geometry(),
        "tasks": [],
    }

    if not args.submit:
        payload["validation"] = {"geometry": "passed", "submission_flag_required": True}
        _write_json(output, payload)
        print(json.dumps(payload, indent=2))
        print(f"Validation manifest written to {output}")
        return

    device = AwsDevice(args.device_arn)
    discretised = programme.discretize(device)
    destination = (args.s3_bucket, args.s3_prefix) if args.s3_bucket else None
    for repetition in range(args.repetitions):
        task = device.run(
            discretised,
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
            record.update({"state": task.state(), **_decode_measurements(result.measurements)})
        payload["tasks"].append(record)
        _write_json(output, payload)
        print(f"Submitted Aquila task {repetition + 1}/{args.repetitions}: {task.id}")

    print(f"Task manifest written to {output}")


if __name__ == "__main__":
    main()

"""Provider-neutral QAOA circuit for the reduced signed benchmark."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from braket.circuits import Circuit

from .problem_instance import REDUCED_INSTANCE, ReducedQUBO


PARAMETER_FILE = Path(__file__).with_name("qaoa_parameters.json")


@dataclass(frozen=True)
class QAOAParameters:
    depth: int
    gamma: tuple[float, ...]
    beta: tuple[float, ...]

    def validate(self) -> None:
        if self.depth not in (1, 2):
            raise ValueError("the registered protocol supports p=1 or p=2")
        if len(self.gamma) != self.depth or len(self.beta) != self.depth:
            raise ValueError("gamma and beta must contain one value per layer")


def load_parameters(depth: int, path: Path = PARAMETER_FILE) -> QAOAParameters:
    payload = json.loads(path.read_text(encoding="utf-8"))
    record = payload[f"p{depth}"]
    parameters = QAOAParameters(
        depth=depth,
        gamma=tuple(float(value) for value in record["gamma_rad"]),
        beta=tuple(float(value) for value in record["beta_rad"]),
    )
    parameters.validate()
    return parameters


def build_qaoa_circuit(
    parameters: QAOAParameters,
    instance: ReducedQUBO = REDUCED_INSTANCE,
    two_qubit_synthesis: str = "zz",
) -> Circuit:
    """Build the scaled Ising QAOA circuit defined in the manuscript."""

    parameters.validate()
    if two_qubit_synthesis not in {"zz", "cnot"}:
        raise ValueError("two_qubit_synthesis must be 'zz' or 'cnot'")
    instance.validate_mapping()
    _, h, J, scale = instance.to_ising()

    circuit = Circuit()
    for qubit in range(instance.n_variables):
        circuit.h(qubit)

    for layer in range(parameters.depth):
        gamma = parameters.gamma[layer]
        beta = parameters.beta[layer]
        for qubit, field in enumerate(h):
            circuit.rz(qubit, 2.0 * gamma * float(field) / scale)
        for i, j, _ in instance.edges:
            angle = 2.0 * gamma * float(J[i, j]) / scale
            if two_qubit_synthesis == "zz":
                circuit.zz(i, j, angle)
            else:
                circuit.cnot(i, j)
                circuit.rz(j, angle)
                circuit.cnot(i, j)
        for qubit in range(instance.n_variables):
            circuit.rx(qubit, 2.0 * beta)
    return circuit


def circuit_metadata(
    parameters: QAOAParameters,
    instance: ReducedQUBO = REDUCED_INSTANCE,
    two_qubit_synthesis: str = "zz",
) -> dict:
    _, h, J, scale = instance.to_ising()
    return {
        "variables": instance.n_variables,
        "parent_jobs": len(set(instance.job_index)),
        "couplings": len(instance.edges),
        "negative_qubo_couplings": sum(weight < 0 for _, _, weight in instance.edges),
        "depth": parameters.depth,
        "logical_zz_rotations": parameters.depth * len(instance.edges),
        "two_qubit_synthesis": two_qubit_synthesis,
        "pre_routing_two_qubit_gates": parameters.depth
        * len(instance.edges)
        * (1 if two_qubit_synthesis == "zz" else 2),
        "gamma_rad": list(parameters.gamma),
        "beta_rad": list(parameters.beta),
        "ising_scale": scale,
        "ising_field_max_abs": float(max(abs(h))),
        "ising_coupling_max_abs": float(max(abs(J.flatten()))),
        "bit_order": "logical qubits 0 to 7, left to right",
    }


def canonical_counts(raw_counts: dict[str, int], n_variables: int) -> dict[str, int]:
    """Normalise Braket count keys to logical-qubit order ``0, ..., n-1``."""

    counts: dict[str, int] = {}
    for raw, count in raw_counts.items():
        key = str(raw).replace(" ", "")
        if len(key) != n_variables or set(key) - {"0", "1"}:
            raise ValueError(f"unexpected measurement key: {raw!r}")
        counts[key] = counts.get(key, 0) + int(count)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

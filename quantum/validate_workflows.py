"""Offline structural checks for all quantum execution workflows."""

from __future__ import annotations

import json

from .aquila_ahs import build_program, validate_geometry
from .problem_instance import REDUCED_INSTANCE
from .qaoa_circuit import build_qaoa_circuit, circuit_metadata, load_parameters


def main() -> None:
    REDUCED_INSTANCE.validate_mapping()
    report = {
        "qubo_to_ising": "passed",
        "exact_ground_energy": min(
            REDUCED_INSTANCE.energy(
                [(mask >> i) & 1 for i in range(REDUCED_INSTANCE.n_variables)]
            )
            for mask in range(1 << REDUCED_INSTANCE.n_variables)
        ),
        "gate_model": {},
        "ahs_geometry": validate_geometry(),
    }
    for depth in (1, 2):
        parameters = load_parameters(depth)
        ionq = build_qaoa_circuit(parameters, two_qubit_synthesis="zz")
        rigetti = build_qaoa_circuit(parameters, two_qubit_synthesis="cnot")
        report["gate_model"][f"p{depth}"] = {
            "logical": circuit_metadata(parameters),
            "ionq_instructions": len(ionq.instructions),
            "rigetti_pre_routing_instructions": len(rigetti.instructions),
        }
    for evolution_us in (2.0, 4.0, 8.0):
        build_program(evolution_us)
    report["ahs_programmes_us"] = [2.0, 4.0, 8.0]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Frozen reduced benchmark and the exact QUBO-to-Ising conversion.

The manuscript's eight-variable benchmark is a binary quadratic objective

    E(x) = sum_i a_i x_i + sum_{i<j} q_ij x_i x_j,

with ``x_i`` in ``{0, 1}``.  Gate-model execution uses ``x_i=(1-Z_i)/2``.
Keeping the binary coefficients as the source of truth prevents the QUBO
linear terms from being mistaken for Pauli-Z fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class ReducedQUBO:
    """A fixed binary quadratic benchmark with scheduling metadata."""

    linear: tuple[float, ...]
    quadratic: tuple[tuple[float, ...], ...]
    job_index: tuple[int, ...]
    weights: tuple[float, ...]
    labels: tuple[str, ...]
    parent_variable_ids: tuple[int, ...]
    power_cap_W: float
    ramp_cap_W_per_s: float
    ramp_weight: float
    parent_seed: int

    @property
    def n_variables(self) -> int:
        return len(self.linear)

    @property
    def matrix(self) -> np.ndarray:
        matrix = np.asarray(self.quadratic, dtype=float)
        if matrix.shape != (self.n_variables, self.n_variables):
            raise ValueError("quadratic matrix has the wrong dimensions")
        if not np.allclose(matrix, matrix.T):
            raise ValueError("quadratic matrix must be symmetric")
        if not np.allclose(np.diag(matrix), 0.0):
            raise ValueError("quadratic matrix diagonal must be zero")
        return matrix

    @property
    def edges(self) -> tuple[tuple[int, int, float], ...]:
        matrix = self.matrix
        return tuple(
            (i, j, float(matrix[i, j]))
            for i in range(self.n_variables)
            for j in range(i + 1, self.n_variables)
            if abs(matrix[i, j]) > 1e-12
        )

    def energy(self, bits: Sequence[int] | str) -> float:
        x = bits_to_array(bits, self.n_variables).astype(float)
        matrix = self.matrix
        return float(np.dot(np.asarray(self.linear), x) + 0.5 * x @ matrix @ x)

    def selected_weight(self, bits: Sequence[int] | str) -> float:
        x = bits_to_array(bits, self.n_variables)
        return float(np.dot(np.asarray(self.weights), x))

    def job_unique(self, bits: Sequence[int] | str) -> bool:
        x = bits_to_array(bits, self.n_variables)
        selected_jobs = [self.job_index[i] for i in np.flatnonzero(x)]
        return len(selected_jobs) == len(set(selected_jobs))

    def to_ising(self) -> tuple[float, np.ndarray, np.ndarray, float]:
        """Return ``constant, h, J, scale`` for ``x=(1-Z)/2``.

        The returned Hamiltonian is

            constant + sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j.

        ``scale`` is the largest absolute non-constant coefficient.  Dividing
        ``h`` and ``J`` by it fixes the dimensionless angle convention used by
        the QAOA circuit without changing any computational-basis minimiser.
        """

        a = np.asarray(self.linear, dtype=float)
        q = self.matrix
        constant = float(0.5 * a.sum() + 0.25 * np.triu(q, 1).sum())
        h = -0.5 * a - 0.25 * q.sum(axis=1)
        J = 0.25 * q
        scale = float(max(np.max(np.abs(h)), np.max(np.abs(J))))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Ising coefficient scale must be positive")
        return constant, h, J, scale

    def validate_mapping(self, atol: float = 1e-10) -> None:
        """Exhaustively verify the binary-to-Ising energy identity."""

        constant, h, J, _ = self.to_ising()
        for mask in range(1 << self.n_variables):
            x = np.array([(mask >> i) & 1 for i in range(self.n_variables)])
            z = 1.0 - 2.0 * x
            mapped = constant + h @ z + 0.5 * z @ J @ z
            if not np.isclose(self.energy(x), mapped, atol=atol, rtol=0.0):
                raise AssertionError(f"QUBO-to-Ising mismatch at mask {mask}")


def bits_to_array(bits: Sequence[int] | str, n_variables: int) -> np.ndarray:
    """Parse a bit string in logical-qubit order ``x_0, ..., x_(n-1)``."""

    if isinstance(bits, str):
        cleaned = bits.replace(" ", "").replace("_", "")
        if set(cleaned) - {"0", "1"}:
            raise ValueError(f"invalid bit string: {bits!r}")
        values: Iterable[int] = (int(value) for value in cleaned)
    else:
        values = bits
    array = np.asarray(list(values), dtype=int)
    if array.shape != (n_variables,) or not np.all(np.isin(array, (0, 1))):
        raise ValueError(f"expected {n_variables} binary values")
    return array


REDUCED_INSTANCE = ReducedQUBO(
    linear=(
        -4.166440992940187,
        -49.46412576717011,
        -28.50396986790942,
        -28.50396986790942,
        -18.56126226542638,
        -8.278099523597207,
        -28.50396986790942,
        -28.50396986790942,
    ),
    quadratic=(
        (0.0, -1.5234108117671785, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (
            -1.5234108117671785,
            0.0,
            23.345911093260455,
            23.503393443352792,
            16.75524190471851,
            11.119520548649055,
            23.345911093260455,
            23.345911093260455,
        ),
        (
            0.0,
            23.345911093260455,
            0.0,
            21.715158900728465,
            12.914832717948645,
            2.233526364522624,
            36.725231119117595,
            36.725231119117595,
        ),
        (
            0.0,
            23.503393443352792,
            21.715158900728465,
            0.0,
            16.735108991106973,
            6.05509800574168,
            21.715158900728465,
            21.715158900728465,
        ),
        (
            0.0,
            16.75524190471851,
            12.914832717948645,
            16.735108991106973,
            0.0,
            7.596862378980445,
            12.914832717948645,
            12.914832717948645,
        ),
        (
            0.0,
            11.119520548649055,
            2.233526364522624,
            6.05509800574168,
            7.596862378980445,
            0.0,
            2.233526364522624,
            2.233526364522624,
        ),
        (
            0.0,
            23.345911093260455,
            36.725231119117595,
            21.715158900728465,
            12.914832717948645,
            2.233526364522624,
            0.0,
            36.725231119117595,
        ),
        (
            0.0,
            23.345911093260455,
            36.725231119117595,
            21.715158900728465,
            12.914832717948645,
            2.233526364522624,
            36.725231119117595,
            0.0,
        ),
    ),
    job_index=(0, 5, 3, 4, 1, 2, 3, 3),
    weights=(
        727450.0,
        727474.0,
        877546.0,
        877546.0,
        877474.0,
        727304.0,
        877546.0,
        877546.0,
    ),
    labels=(
        "r200/o256",
        "r10/o256",
        "r20/o1024",
        "r20/o1024",
        "r50/o1024",
        "r50/o256",
        "r20/o1024",
        "r20/o1024",
    ),
    parent_variable_ids=(4, 155, 90, 120, 30, 60, 91, 92),
    power_cap_W=22_000.0,
    ramp_cap_W_per_s=9_000.0,
    ramp_weight=21.5,
    parent_seed=0,
)

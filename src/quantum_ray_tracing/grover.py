"""Grover search, BBHT exponential search (QSearch) and Duerr-Hoyer minimum finding.

This module replicates the intersection machinery of

    L. P. Santos, T. Bashford-Rogers, J. Barbosa, P. Navratil,
    "Towards Quantum Ray Tracing", arXiv:2204.12797 (2022).

A ray/scene query is phrased as a search over ``N = 2**n`` primitive indices.  The
oracle marks every index ``p`` for which a classical predicate holds (``ray hits
primitive p``, optionally ``and closer than the current best``).  As in the paper,
the per-primitive predicate is evaluated classically when the oracle is built
(Santos et al. also compute their ray/plane coordinates classically, Sec. 4); the
quantity whose scaling we measure is the number of *oracle evaluations* (Grover
iterations) plus the classical verifications of measured candidates, exactly the
paper's ``#Int = #Eval + #C_Int`` metric.

Two interchangeable backends execute a Grover circuit and return one measurement:

* :class:`AerGroverBackend` builds the actual circuit (phase oracle + diffuser) and
  runs it on Qiskit Aer with ``shots=1``.
* :class:`ExactGroverBackend` samples from the exact output distribution of the
  ideal circuit, which lives in the 2-D span of the marked/unmarked uniform states:
  ``P(marked) = sin^2((2r+1) theta)``, ``sin^2 theta = M/N``.  The test-suite checks
  this against Aer, and it lets us render full images in seconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import numpy as np
from qiskit import QuantumCircuit

__all__ = [
    "phase_oracle",
    "diffuser",
    "grover_circuit",
    "grover_success_probability",
    "GroverBackend",
    "ExactGroverBackend",
    "AerGroverBackend",
    "QueryCounter",
    "qsearch",
    "minimum_finding",
]


# --------------------------------------------------------------------------- circuits
def _mcz(qc: QuantumCircuit, qubits: Sequence[int]) -> None:
    """Multi-controlled Z on ``qubits`` (symmetric, so no designated target)."""
    qubits = list(qubits)
    if len(qubits) == 1:
        qc.z(qubits[0])
        return
    target = qubits[-1]
    qc.h(target)
    qc.mcx(qubits[:-1], target)
    qc.h(target)


def phase_oracle(n: int, marked: Sequence[int]) -> QuantumCircuit:
    """Phase oracle ``O|x> = (-1)^{[x in marked]} |x>`` on ``n`` qubits (little-endian)."""
    qc = QuantumCircuit(n, name="Oracle")
    for m in sorted(set(int(x) for x in marked)):
        if not 0 <= m < 2**n:
            raise ValueError(f"marked index {m} out of range for {n} qubits")
        zeros = [q for q in range(n) if not (m >> q) & 1]
        if zeros:
            qc.x(zeros)
        _mcz(qc, range(n))
        if zeros:
            qc.x(zeros)
    return qc


def diffuser(n: int) -> QuantumCircuit:
    """Grover diffusion operator ``2|s><s| - I`` (up to global phase)."""
    qc = QuantumCircuit(n, name="Diffuser")
    qc.h(range(n))
    qc.x(range(n))
    _mcz(qc, range(n))
    qc.x(range(n))
    qc.h(range(n))
    return qc


def grover_circuit(n: int, marked: Sequence[int], iterations: int, measure: bool = True) -> QuantumCircuit:
    """``H^n`` followed by ``iterations`` Grover iterations, optionally measured."""
    qc = QuantumCircuit(n, n if measure else 0)
    qc.h(range(n))
    oracle = phase_oracle(n, marked)
    diff = diffuser(n)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diff, inplace=True)
    if measure:
        qc.measure(range(n), range(n))
    return qc


def grover_success_probability(n_items: int, n_marked: int, iterations: int) -> float:
    """Probability of measuring *some* marked item after ``iterations`` iterations."""
    if n_marked <= 0:
        return 0.0
    if n_marked >= n_items:
        return 1.0
    theta = math.asin(math.sqrt(n_marked / n_items))
    return math.sin((2 * iterations + 1) * theta) ** 2


# --------------------------------------------------------------------------- backends
class GroverBackend(Protocol):
    def sample(self, n: int, marked: np.ndarray, iterations: int) -> int:
        """Run one Grover circuit and return the measured index."""


class ExactGroverBackend:
    """Samples the exact measurement distribution of the ideal Grover circuit."""

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng if rng is not None else np.random.default_rng()

    def sample(self, n: int, marked: np.ndarray, iterations: int) -> int:
        N = 2**n
        marked = np.asarray(marked, dtype=np.int64)
        M = marked.size
        if M == 0 or M == N:
            return int(self.rng.integers(N))
        p_good = grover_success_probability(N, M, iterations)
        if self.rng.random() < p_good:
            return int(marked[self.rng.integers(M)])
        # uniform over the unmarked items
        mask = np.ones(N, dtype=bool)
        mask[marked] = False
        unmarked = np.flatnonzero(mask)
        return int(unmarked[self.rng.integers(unmarked.size)])


class AerGroverBackend:
    """Executes the real Grover circuit on Qiskit Aer (one shot per call)."""

    def __init__(self, seed: int | None = None):
        from qiskit_aer import AerSimulator

        self.sim = AerSimulator(seed_simulator=seed)
        self.circuits_run = 0

    def sample(self, n: int, marked: np.ndarray, iterations: int) -> int:
        qc = grover_circuit(n, list(np.asarray(marked).tolist()), iterations)
        # H, X, MCX and measure are native Aer instructions, so no transpilation is
        # needed (this is ~50x faster than transpiling every per-ray circuit).
        counts = self.sim.run(qc, shots=1).result().get_counts()
        self.circuits_run += 1
        (bitstring,) = counts.keys()
        return int(bitstring, 2)


# --------------------------------------------------------------------------- algorithms
@dataclass
class QueryCounter:
    """Cost bookkeeping in the units of Santos et al. (Sec. 6)."""

    oracle_calls: int = 0  # #Eval: Grover iterations (each = one oracle application)
    classical_checks: int = 0  # #C_Int: classical verification of a measured index
    circuits: int = 0  # number of Grover circuits executed / measured

    @property
    def total(self) -> int:
        return self.oracle_calls + self.classical_checks

    def __iadd__(self, other: "QueryCounter") -> "QueryCounter":
        self.oracle_calls += other.oracle_calls
        self.classical_checks += other.classical_checks
        self.circuits += other.circuits
        return self


def qsearch(
    predicate: np.ndarray,
    backend: GroverBackend,
    rng: np.random.Generator,
    counter: QueryCounter | None = None,
    c: float = 1.2,
    max_rounds: int | None = None,
) -> int | None:
    """Exponential Grover search for an unknown number of solutions (BBHT 1998).

    This is Algorithm 1 ("QSearch") of Santos et al.: try one random classical
    sample, then for ``l = 1, 2, ...`` run ``Q^r A`` with ``r`` drawn uniformly from
    ``[0, M_l)``, ``M_l = min(c^l, sqrt(N))``, and verify the measured candidate
    classically.  ``predicate`` is a boolean array over the ``N = 2**n`` indices; it
    defines the oracle and the classical check.  Returns a solution index or
    ``None`` when the round budget is exhausted (the probabilistic miss that causes
    the noise analysed in the paper).
    """
    counter = counter if counter is not None else QueryCounter()
    predicate = np.asarray(predicate, dtype=bool)
    N = predicate.size
    n = int(round(math.log2(N)))
    if 2**n != N:
        raise ValueError("predicate size must be a power of two")
    marked = np.flatnonzero(predicate)
    sqrtN = math.sqrt(N)
    if max_rounds is None:
        # enough rounds for M_l to saturate at sqrt(N), then a few more trials there.
        max_rounds = int(math.ceil(math.log(sqrtN, c))) + 4 if sqrtN > 1 else 4

    # random classical guess (one ray/primitive intersection test)
    x = int(rng.integers(N))
    counter.classical_checks += 1
    if predicate[x]:
        return x
    for l in range(1, max_rounds + 1):
        M_l = min(c**l, sqrtN)
        r = int(rng.integers(0, max(1, int(math.ceil(M_l)))))
        x = backend.sample(n, marked, r)
        counter.oracle_calls += r
        counter.circuits += 1
        counter.classical_checks += 1
        if predicate[x]:
            return x
    return None


def minimum_finding(
    values: np.ndarray,
    valid: np.ndarray,
    backend: GroverBackend,
    rng: np.random.Generator,
    counter: QueryCounter | None = None,
    c: float = 1.2,
    max_rounds: int | None = None,
) -> int | None:
    """Duerr-Hoyer quantum minimum finding over the ``valid`` entries of ``values``.

    Implements the closest-hit search (``QTrace``/``TracePass``, Algs. 2-4 of Santos
    et al.): repeatedly QSearch for an index with ``valid & values < current`` and
    stop when a search fails.  Returns the index of the (probably) minimal value or
    ``None`` if no valid index was found at all.
    """
    counter = counter if counter is not None else QueryCounter()
    values = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    best = qsearch(valid, backend, rng, counter, c=c, max_rounds=max_rounds)
    if best is None:
        return None
    while True:
        # When nothing is better the algorithm cannot know it: it has to run (and pay
        # for) a full failing search before it stops.
        better = valid & (values < values[best])
        cand = qsearch(better, backend, rng, counter, c=c, max_rounds=max_rounds)
        if cand is None:
            return best
        best = cand

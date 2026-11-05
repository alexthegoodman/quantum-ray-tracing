"""Quantum amplitude estimation without phase estimation (quantum numerical integration).

Mosier, McGuire & Hachisuka (SIGGRAPH Asia 2023) estimate the path integral encoded
in a single qubit ``|F'> = sqrt(1-a)|0> + sqrt(a)|1>`` with an amplitude-estimation
method "based on amplitude amplification instead of phase estimation" (Nakaji 2020;
Shimada & Hachisuka 2020).  We provide two estimators from that family, both built
on ``a = sin^2(theta)`` and ``P(good | m) = sin^2((2m+1) theta)``:

* :func:`iqae` - Iterative Amplitude Estimation (Grinko et al. 2021), our default.
  Like Nakaji's Faster AE used by the paper, it maintains a confidence interval
  for ``theta`` and only chooses Grover powers whose phase cannot alias, so the
  error is bounded with probability ``1 - alpha``.
* :func:`mlae` - Maximum-Likelihood AE (Suzuki et al. 2020) with an exponential
  schedule.  Its *median* error scales as ``1/N`` but rare alias failures make
  its RMSE erratic at small shot counts.

Cost is measured in *calls of A* (the ray-marching circuit), which is the paper's
metric "the number of times our ray marching circuit was executed":
one shot of ``Q^m A`` costs ``2m + 1`` calls.  Classical Monte-Carlo costs one call
per sampled path.  MLAE error scales as ``O(1/N)`` versus ``O(1/sqrt N)`` for MC.

Samplers
--------
* :class:`AerAmplitudeSampler` - builds ``Q^m A`` from a state-preparation circuit,
  runs it on Qiskit Aer and measures the objective qubit.
* :class:`ExactAmplitudeSampler` - draws binomial counts from the exact ideal law
  ``sin^2((2m+1) theta)``; validated against Aer in the tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import grover_operator

__all__ = [
    "amplification_operator",
    "AmplitudeSampler",
    "ExactAmplitudeSampler",
    "AerAmplitudeSampler",
    "MLAEResult",
    "mlae",
    "default_schedule",
    "mlae_queries",
    "IQAEResult",
    "iqae",
]


def amplification_operator(state_preparation: QuantumCircuit, objective_qubit: int) -> QuantumCircuit:
    """Grover operator ``Q = A S_0 A^dagger S_chi`` whose good states have objective=1."""
    n = state_preparation.num_qubits
    oracle = QuantumCircuit(n, name="S_chi")
    oracle.z(objective_qubit)
    return grover_operator(oracle, state_preparation=state_preparation)


class AmplitudeSampler(Protocol):
    def counts(self, power: int, shots: int) -> int:
        """Number of 'good' outcomes in ``shots`` measurements of ``Q^power A``."""


class ExactAmplitudeSampler:
    """Ideal-circuit statistics for a known amplitude ``a``."""

    def __init__(self, a: float, rng: np.random.Generator | None = None):
        if not -1e-12 <= a <= 1 + 1e-12:
            raise ValueError("amplitude must be a probability")
        self.a = float(min(max(a, 0.0), 1.0))
        self.theta = math.asin(math.sqrt(self.a))
        self.rng = rng if rng is not None else np.random.default_rng()

    def probability(self, power: int) -> float:
        return math.sin((2 * power + 1) * self.theta) ** 2

    def counts(self, power: int, shots: int) -> int:
        return int(self.rng.binomial(shots, self.probability(power)))


class AerAmplitudeSampler:
    """Runs ``Q^m A`` on Qiskit Aer and counts objective-qubit = 1 outcomes."""

    def __init__(self, state_preparation: QuantumCircuit, objective_qubit: int, seed: int | None = None):
        from qiskit_aer import AerSimulator

        self.A = state_preparation
        self.obj = objective_qubit
        self.Q = amplification_operator(state_preparation, objective_qubit)
        self.sim = AerSimulator()
        self.rng = np.random.default_rng(seed)
        # Transpile A and Q once; Q^m A is then assembled from the compiled pieces.
        self._A = self._compile(self.A)
        self._Q = self._compile(self.Q)
        self._cache: dict[int, QuantumCircuit] = {}

    def _compile(self, qc: QuantumCircuit) -> QuantumCircuit:
        """Transpile for Aer, expanding Aer's ``multiplexer`` instruction.

        Aer 0.18 segfaults when it runs a circuit assembled from several compiled
        pieces that contain ``multiplexer``; plain gates avoid that.
        """
        t = transpile(qc, self.sim, optimization_level=1)
        while "multiplexer" in t.count_ops():
            t = transpile(t.decompose(gates_to_decompose=["multiplexer"]), self.sim, optimization_level=0)
        return t

    def _body(self, power: int, n_clbits: int) -> QuantumCircuit:
        qc = QuantumCircuit(self.A.num_qubits, n_clbits)
        qc.compose(self._A, inplace=True)
        for _ in range(power):
            qc.compose(self._Q, inplace=True)
        return qc

    def circuit(self, power: int) -> QuantumCircuit:
        if power not in self._cache:
            qc = self._body(power, 1)
            qc.measure(self.obj, 0)
            self._cache[power] = qc
        return self._cache[power]

    def probability(self, power: int) -> float:
        """Exact probability of objective=1 after ``Q^power A`` via Aer statevector."""
        qc = self._body(power, 0)
        qc.save_probabilities([self.obj])
        probs = self.sim.run(qc, shots=1).result().data()["probabilities"]
        return float(probs[1])

    def counts(self, power: int, shots: int) -> int:
        seed = int(self.rng.integers(2**31))
        res = self.sim.run(self.circuit(power), shots=shots, seed_simulator=seed).result().get_counts()
        return int(res.get("1", 0))


@dataclass
class MLAEResult:
    estimate: float
    theta: float
    powers: list[int]
    hits: list[int]
    shots: int
    queries: int  # total calls of A


def default_schedule(k: int) -> list[int]:
    """Exponential schedule ``[0, 1, 2, 4, ..., 2^(k-2)]`` (``k`` circuits)."""
    return [0] + [2**j for j in range(k - 1)]


def mlae_queries(powers: Sequence[int], shots: int) -> int:
    return int(shots * sum(2 * m + 1 for m in powers))


def _log_likelihood(theta: np.ndarray, powers: Sequence[int], hits: Sequence[int], shots: int) -> np.ndarray:
    ll = np.zeros_like(theta)
    eps = 1e-300
    for m, h in zip(powers, hits):
        p = np.sin((2 * m + 1) * theta) ** 2
        ll += h * np.log(np.maximum(p, eps)) + (shots - h) * np.log(np.maximum(1 - p, eps))
    return ll


def mlae(sampler: AmplitudeSampler, powers: Sequence[int], shots: int) -> MLAEResult:
    """Maximum-likelihood amplitude estimation with the given Grover-power schedule."""
    powers = list(powers)
    hits = [sampler.counts(m, shots) for m in powers]
    # Global grid search (the likelihood is multimodal), then a fine local search.
    m_max = max(powers)
    n_grid = max(2000, 200 * (2 * m_max + 1))
    grid = np.linspace(0.0, math.pi / 2, n_grid)
    ll = _log_likelihood(grid, powers, hits, shots)
    i = int(np.argmax(ll))
    step = grid[1] - grid[0]
    lo, hi = max(0.0, grid[i] - step), min(math.pi / 2, grid[i] + step)
    for _ in range(3):
        fine = np.linspace(lo, hi, 201)
        j = int(np.argmax(_log_likelihood(fine, powers, hits, shots)))
        step = fine[1] - fine[0]
        lo, hi = max(0.0, fine[j] - step), min(math.pi / 2, fine[j] + step)
    theta = float(fine[j])
    return MLAEResult(
        estimate=math.sin(theta) ** 2,
        theta=theta,
        powers=powers,
        hits=hits,
        shots=shots,
        queries=mlae_queries(powers, shots),
    )


# --------------------------------------------------------------------------- IQAE
@dataclass
class IQAEResult:
    estimate: float
    interval: tuple[float, float]
    powers: list[int]
    queries: int  # total calls of A
    rounds: int


def _find_next_k(k: int, theta_l: float, theta_u: float, up: bool, r: float = 2.0) -> tuple[int, bool]:
    """Largest power ``k'`` with ``(4k'+2) [theta_l, theta_u]`` inside one half-circle."""
    K_i = 4 * k + 2
    width = theta_u - theta_l
    K_max = int(math.floor(math.pi / width)) if width > 0 else 4 * 10**9
    K = K_max - (K_max - 2) % 4
    while K >= r * K_i:
        lo, hi = K * theta_l, K * theta_u
        if math.floor(lo / math.pi) == math.floor(hi / math.pi) or hi % math.pi == 0:
            half_up = (lo % (2 * math.pi)) < math.pi or math.isclose(lo % (2 * math.pi), math.pi)
            return (K - 2) // 4, half_up
        K -= 4
    return k, up


def iqae(
    sampler: AmplitudeSampler,
    epsilon: float,
    alpha: float = 0.05,
    shots: int = 64,
    max_rounds: int = 200,
) -> IQAEResult:
    """Iterative quantum amplitude estimation with Chernoff-Hoeffding intervals.

    Returns an estimate whose ``theta`` interval has half-width ``<= epsilon``
    with probability ``>= 1 - alpha`` (Grinko, Gacon, Zoufal & Woerner 2021).
    """
    T = max(1, int(math.ceil(math.log2(math.pi / (8 * epsilon)))))
    theta_l, theta_u = 0.0, math.pi / 2
    k, up = 0, True
    powers: list[int] = []
    queries = 0
    hits_k, n_k = 0, 0
    rounds = 0
    while theta_u - theta_l > 2 * epsilon and rounds < max_rounds:
        rounds += 1
        k_new, up = _find_next_k(k, theta_l, theta_u, up)
        if k_new != k:
            hits_k, n_k = 0, 0  # restart the running counts for the new power
        k = k_new
        h = sampler.counts(k, shots)
        powers.append(k)
        queries += shots * (2 * k + 1)
        hits_k += h
        n_k += shots
        p = hits_k / n_k
        eps_a = math.sqrt(math.log(2 * T / alpha) / (2 * n_k))
        a_min, a_max = max(0.0, p - eps_a), min(1.0, p + eps_a)
        K = 4 * k + 2
        if up:
            t_min, t_max = math.acos(1 - 2 * a_min), math.acos(1 - 2 * a_max)
        else:
            t_min, t_max = 2 * math.pi - math.acos(1 - 2 * a_max), 2 * math.pi - math.acos(1 - 2 * a_min)
        base_l = 2 * math.pi * math.floor(K * theta_l / (2 * math.pi))
        base_u = 2 * math.pi * math.floor(K * theta_u / (2 * math.pi))
        new_l = (base_l + t_min) / K
        new_u = (base_u + t_max) / K
        # never widen the interval; guard against numerical crossings
        theta_l, theta_u = max(theta_l, min(new_l, new_u)), min(theta_u, max(new_l, new_u))
        if theta_l > theta_u:
            theta_l = theta_u = 0.5 * (theta_l + theta_u)
    a_l, a_u = math.sin(theta_l) ** 2, math.sin(theta_u) ** 2
    return IQAEResult(estimate=0.5 * (a_l + a_u), interval=(a_l, a_u), powers=powers, queries=queries, rounds=rounds)

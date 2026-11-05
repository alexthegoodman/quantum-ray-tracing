import math

import numpy as np
import pytest
from qiskit import QuantumCircuit

from quantum_ray_tracing.amplitude_estimation import (
    AerAmplitudeSampler,
    ExactAmplitudeSampler,
    default_schedule,
    mlae,
    mlae_queries,
)


def _toy_state_prep():
    A = QuantumCircuit(3)
    A.h(0)
    A.ry(0.9, 1)
    A.ccx(0, 1, 2)
    A.cry(0.4, 0, 2)
    return A


def test_amplification_matches_ideal_law():
    A = _toy_state_prep()
    s = AerAmplitudeSampler(A, 2, seed=0)
    a = s.probability(0)
    ex = ExactAmplitudeSampler(a)
    for m in range(6):
        assert s.probability(m) == pytest.approx(ex.probability(m), abs=1e-10)


def test_aer_counts_follow_probability():
    A = _toy_state_prep()
    s = AerAmplitudeSampler(A, 2, seed=4)
    p = s.probability(2)
    assert s.counts(2, 4000) / 4000 == pytest.approx(p, abs=0.03)


@pytest.mark.parametrize("a", [0.0, 0.03, 0.31, 0.5, 0.97])
def test_mlae_accuracy(a):
    rng = np.random.default_rng(7)
    errs = [mlae(ExactAmplitudeSampler(a, rng), default_schedule(8), 20).estimate - a for _ in range(30)]
    assert np.sqrt(np.mean(np.square(errs))) < 3e-3


def test_mlae_on_aer_circuits():
    A = _toy_state_prep()
    s = AerAmplitudeSampler(A, 2, seed=5)
    a = s.probability(0)
    est = np.mean([mlae(s, default_schedule(5), 30).estimate for _ in range(4)])
    assert est == pytest.approx(a, abs=0.01)


def test_heisenberg_scaling_beats_monte_carlo():
    """In the asymptotic regime MLAE error falls like 1/N (Monte Carlo: 1/sqrt N)."""
    rng = np.random.default_rng(3)
    a = 0.2
    Ks = (6, 8, 10)
    cost, rmse = [], []
    for K in Ks:
        errs = [mlae(ExactAmplitudeSampler(a, rng), default_schedule(K), 32).estimate - a for _ in range(200)]
        rmse.append(np.sqrt(np.mean(np.square(errs))))
        cost.append(mlae_queries(default_schedule(K), 32))
    slope = np.polyfit(np.log(cost), np.log(rmse), 1)[0]
    assert slope < -0.8  # Monte Carlo would give -0.5


# ------------------------------------------------------------------------ IQAE
from quantum_ray_tracing.amplitude_estimation import iqae


@pytest.mark.parametrize("a", [0.0, 0.02, 0.25, 0.5, 0.8, 1.0])
def test_iqae_interval_covers_truth(a):
    rng = np.random.default_rng(11)
    runs = [iqae(ExactAmplitudeSampler(a, rng), 1e-3, alpha=0.01, shots=32) for _ in range(100)]
    covered = np.mean([r.interval[0] - 1e-12 <= a <= r.interval[1] + 1e-12 for r in runs])
    assert covered >= 0.97
    assert all(r.interval[1] - r.interval[0] <= 2.1e-3 * 2 for r in runs)


def test_iqae_error_scales_like_one_over_n():
    rng = np.random.default_rng(12)
    a = 0.3
    cost, rmse = [], []
    for eps in (1e-2, 1e-3, 1e-4):
        runs = [iqae(ExactAmplitudeSampler(a, rng), eps, alpha=1e-3, shots=32) for _ in range(150)]
        cost.append(np.mean([r.queries for r in runs]))
        rmse.append(np.sqrt(np.mean([(r.estimate - a) ** 2 for r in runs])))
    slope = np.polyfit(np.log(cost), np.log(rmse), 1)[0]
    assert -1.3 < slope < -0.8


def test_iqae_on_aer_circuits():
    A = _toy_state_prep()
    s = AerAmplitudeSampler(A, 2, seed=8)
    a = s.probability(0)
    r = iqae(s, 1e-3, alpha=1e-3, shots=32)
    assert r.interval[0] - 1e-9 <= a <= r.interval[1] + 1e-9
    assert max(r.powers) > 50  # really used deep Grover powers

import math

import numpy as np
import pytest
from qiskit import transpile

from quantum_ray_tracing.grover import (
    AerGroverBackend,
    ExactGroverBackend,
    QueryCounter,
    grover_circuit,
    grover_success_probability,
    minimum_finding,
    qsearch,
)


@pytest.mark.parametrize("n,marked,r", [(3, [5], 2), (4, [0, 9], 1), (5, [3], 4), (5, [1, 7, 30], 2), (6, [17], 6)])
def test_grover_circuit_matches_closed_form(sim, n, marked, r):
    qc = grover_circuit(n, marked, r, measure=False)
    qc.save_probabilities()
    p = np.asarray(sim.run(transpile(qc, sim)).result().data()["probabilities"])
    expected_good = grover_success_probability(2**n, len(marked), r)
    assert p[marked].sum() == pytest.approx(expected_good, abs=1e-9)
    # amplitude is uniform within the marked and within the unmarked sets
    assert np.allclose(p[marked], expected_good / len(marked))
    unmarked = np.setdiff1d(np.arange(2**n), marked)
    assert np.allclose(p[unmarked], (1 - expected_good) / unmarked.size)


def test_exact_backend_distribution():
    rng = np.random.default_rng(0)
    be = ExactGroverBackend(rng)
    marked = np.array([2, 11])
    draws = np.array([be.sample(4, marked, 1) for _ in range(20000)])
    p_good = grover_success_probability(16, 2, 1)
    assert np.isin(draws, marked).mean() == pytest.approx(p_good, abs=0.01)


def test_aer_backend_finds_single_marked_item():
    be = AerGroverBackend(seed=3)
    # 3 iterations on 16 items gives 96% success
    hits = sum(be.sample(4, np.array([6]), 3) == 6 for _ in range(30))
    assert hits >= 25


@pytest.mark.parametrize("backend_cls", ["exact", "aer"])
def test_qsearch_and_minimum_finding(backend_cls):
    rng = np.random.default_rng(1)
    be = ExactGroverBackend(rng) if backend_cls == "exact" else AerGroverBackend(seed=1)
    values = rng.permutation(32).astype(float)
    valid = np.zeros(32, bool)
    valid[[3, 8, 20, 27]] = True
    found = [minimum_finding(values, valid, be, rng) for _ in range(40 if backend_cls == "exact" else 10)]
    truth = np.flatnonzero(valid)[np.argmin(values[valid])]
    assert np.mean([f == truth for f in found]) >= 0.85
    assert qsearch(np.zeros(32, bool), be, rng) is None


def test_qsearch_query_cost_scales_like_sqrt_n():
    rng = np.random.default_rng(2)
    costs = {}
    for n in (6, 10):
        N = 2**n
        cnt = QueryCounter()
        for _ in range(300):
            pred = np.zeros(N, bool)
            pred[rng.integers(N)] = True
            qsearch(pred, ExactGroverBackend(rng), rng, cnt)
        costs[N] = cnt.total / 300
    ratio = costs[1024] / costs[64]
    assert 2.5 < ratio < 6.0  # sqrt(16) = 4; classical would be 16
    assert costs[1024] < 1024 / 4

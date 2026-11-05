import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile

from quantum_ray_tracing.qrm import (
    DIRS,
    Scene2D,
    _shift,
    build_qrm_circuit,
    cornell_2d,
    enumerate_paths,
    exact_radiance,
    sample_paths,
    start_direction_weights,
)


def _objective_probability(sim, q):
    qc = q.circuit.copy()
    qc.save_probabilities([q.objective])
    return sim.run(transpile(qc, sim, optimization_level=1)).result().data()["probabilities"][1]


@pytest.mark.parametrize("d", range(8))
def test_shift_moves_walker_on_basis_states(sim, d):
    for x, y in [(0, 0), (3, 4), (7, 7), (5, 1)]:
        qc = QuantumCircuit(9, 9)
        for j in range(3):
            if (x >> j) & 1:
                qc.x(j)
            if (y >> j) & 1:
                qc.x(3 + j)
            if (d >> j) & 1:
                qc.x(6 + j)
        _shift(qc, qc.qubits[0:3], qc.qubits[3:6], qc.qubits[6:9])
        qc.measure(range(9), range(9))
        (bits,) = sim.run(transpile(qc, sim), shots=1).result().get_counts()
        v = int(bits, 2)
        assert (v & 7, (v >> 3) & 7, v >> 6) == ((x + DIRS[d, 0]) % 8, (y + DIRS[d, 1]) % 8, d)


def test_scene_is_closed():
    """Every voxel reachable from inside is bounded by surfaces whose lobes point inwards."""
    sc = cornell_2d()
    W = sc.size
    for x in range(W):
        for y in range(W):
            n = sc.normal[x, y]
            if n < 0:
                continue
            for d in np.flatnonzero(start_direction_weights(sc, x, y)):
                nx, ny = x + DIRS[d, 0], y + DIRS[d, 1]
                assert 0 <= nx < W and 0 <= ny < W, (x, y, d)


def test_lobe_weights():
    sc = cornell_2d()
    w = start_direction_weights(sc, 3, 0)  # floor, normal north
    assert w[2] == pytest.approx(0.5) and w[1] == pytest.approx(0.25) and w[3] == pytest.approx(0.25)
    assert start_direction_weights(sc, 3, 3) == pytest.approx(np.full(8, 1 / 8))


@pytest.mark.parametrize("steps", [1, 2])
def test_circuit_equals_exact_path_integral_cornell(sim, steps):
    sc = cornell_2d()
    for (x, y) in [(3, 5), (4, 6), (2, 1), (0, 4), (6, 6)]:
        for ch in (0, 1):
            q = build_qrm_circuit(sc, steps, ch, (x, y))
            f, w = enumerate_paths(sc, x, y, steps)
            assert q.scale * _objective_probability(sim, q) == pytest.approx(f[ch] @ w, abs=1e-9)


def _random_scene(seed):
    rng = np.random.default_rng(seed)
    base = cornell_2d()
    return Scene2D(3, base.normal.copy(), rng.uniform(0, 1, base.le.shape), rng.uniform(0, 1, base.rho.shape))


@pytest.mark.parametrize("seed", [0, 1])
def test_circuit_equals_exact_path_integral_random_materials(sim, seed):
    """Random Le/rho everywhere: every Sample/Path term and product is exercised."""
    sc = _random_scene(seed)
    rng = np.random.default_rng(seed)
    for _ in range(4):
        x, y = rng.integers(0, 8, size=2)
        q = build_qrm_circuit(sc, 2, 0, (int(x), int(y)))
        f, w = enumerate_paths(sc, int(x), int(y), 2)
        assert q.scale * _objective_probability(sim, q) == pytest.approx(f[0] @ w, abs=1e-9)


@pytest.mark.slow
@pytest.mark.parametrize("steps", [3, 4])
def test_deep_walk_with_mps(mps, steps):
    sc = _random_scene(5)
    for (x, y) in [(2, 5), (6, 3)]:
        q = build_qrm_circuit(sc, steps, 0, (x, y))
        f, w = enumerate_paths(sc, x, y, steps)
        assert q.scale * _objective_probability(mps, q) == pytest.approx(f[0] @ w, abs=1e-7)


def test_monte_carlo_is_unbiased():
    sc = cornell_2d()
    rng = np.random.default_rng(0)
    f, w = enumerate_paths(sc, 2, 5, 3)
    mc = sample_paths(sc, 2, 5, 3, 40000, rng)[0].mean()
    assert mc == pytest.approx(f[0] @ w, abs=0.01)


def test_light_map_energy_bounds():
    sc = cornell_2d()
    L, F = exact_radiance(sc, 4)
    assert (F >= 0).all() and (F <= 4).all()
    # the light patch itself is the brightest region
    assert L[0, 3, 7] == L[0].max()


def test_iqae_on_real_qrm_circuit():
    """End to end: quantum ray marching circuit + Grover powers on Aer + IQAE."""
    from quantum_ray_tracing.amplitude_estimation import AerAmplitudeSampler, iqae

    sc = cornell_2d()
    q = build_qrm_circuit(sc, 1, 0, (2, 6))
    f, w = enumerate_paths(sc, 2, 6, 1)
    r = iqae(AerAmplitudeSampler(q.circuit, q.objective, seed=3), 1e-3, alpha=1e-3, shots=32)
    assert q.scale * r.interval[0] - 1e-9 <= f[0] @ w <= q.scale * r.interval[1] + 1e-9

import itertools
from dataclasses import replace

import numpy as np
import pytest
from qiskit import QuantumCircuit, QuantumRegister

from quantum_ray_tracing.amplitude_estimation import AerAmplitudeSampler, ExactAmplitudeSampler, iqae
from quantum_ray_tracing.grover import ExactGroverBackend, QueryCounter, minimum_finding, qsearch
from quantum_ray_tracing.particles import (
    ParticleSystem,
    add_constant,
    add_register,
    build_qps_circuit,
    exact_amplitude,
    exact_image,
    firework,
    fountain,
    sample_image,
    simulate,
    simulate_permutation,
    tiny,
)


def _all_states(n):
    return np.array(list(itertools.product([0, 1], repeat=n)), dtype=bool)[:, ::-1]


def _val(st, idx):
    return sum(st[:, q].astype(int) << j for j, q in enumerate(idx))


# ------------------------------------------------------------------ arithmetic
@pytest.mark.parametrize("c", [1, 3, -1, -5, 6, 0])
def test_add_constant_is_modular_addition(c):
    r, k = QuantumRegister(4, "r"), QuantumRegister(1, "k")
    qc = QuantumCircuit(r, k)
    add_constant(qc, r, c, [k[0]])
    st = _all_states(5)
    out = simulate_permutation(qc, st)
    x, ctl = _val(st, range(4)), st[:, 4]
    assert np.array_equal(_val(out, range(4)), np.where(ctl, (x + c) % 16, x))


def test_add_register_signed_source():
    t, s = QuantumRegister(5, "t"), QuantumRegister(3, "s")
    qc = QuantumCircuit(t, s)
    add_register(qc, t, s)
    st = _all_states(8)
    out = simulate_permutation(qc, st)
    sv = _val(st, range(5, 8))
    sv = np.where(sv >= 4, sv - 8, sv)  # two's complement
    assert np.array_equal(_val(out, range(5)), (_val(st, range(5)) + sv) % 32)
    assert np.array_equal(out[:, 5:], st[:, 5:])


def test_permutation_simulator_matches_aer(sim):
    """The gate-level classical simulator agrees with Aer on basis states."""
    from qiskit import transpile

    q = build_qps_circuit(tiny())
    rng = np.random.default_rng(0)
    seeds = rng.integers(0, tiny().num_particles, 3)
    st = simulate_permutation(q.dynamics, q.seeds_to_states(seeds))
    for s0, row in zip(seeds, st):
        qc = QuantumCircuit(q.num_qubits, q.num_qubits)
        for j, qb in enumerate(q.seed_qubits):
            if (s0 >> j) & 1:
                qc.x(qb)
        qc.compose(q.dynamics, inplace=True)
        qc.measure(range(q.num_qubits), range(q.num_qubits))
        (bits,) = sim.run(transpile(qc, sim), shots=1).result().get_counts()
        assert [b == "1" for b in bits[::-1]] == row.tolist()


# ------------------------------------------------------------------ circuit == particle integrator
SMALL_STREAM = ParticleSystem(
    name="small-stream",
    velocities=((-2, 5), (-1, 6), (1, 6), (2, 5)),
    origin=(14, 0),
    steps=7,
    stream=True,
    screen_bits=3,
    sub_bits=2,
    vel_bits=5,
    spread_bits=(1, 0),
    gravity=1,
    kicks_x=(-1, 1),
    kicks_y=(0, 1),
    kick_target="position",
    bounce=True,
    bounce_loss=1,
    margin_bits=2,
    palette="water",
)


def test_circuit_equals_integrator_exhaustively():
    """Every one of the 2^20 particles of a stream with bounce, jitter and ages."""
    s = SMALL_STREAM
    assert s.seed_bits == 20
    q = build_qps_circuit(s)
    seeds = np.arange(s.num_particles)
    st = simulate_permutation(q.dynamics, q.seeds_to_states(seeds))
    r = simulate(s, seeds)
    for reg, signed in (("x", False), ("y", False), ("vx", True), ("vy", True)):
        assert np.array_equal(q.read(st, reg, signed), r[reg]), reg
    assert np.array_equal(q.read(st, "age"), r["age"])


@pytest.mark.parametrize("system", [firework(12), fountain(), firework(5)], ids=["firework12", "fountain", "firework5"])
def test_circuit_equals_integrator_on_random_particles(system):
    q = build_qps_circuit(system)
    seeds = system.random_seeds(3000, np.random.default_rng(1))
    st = simulate_permutation(q.dynamics, q.seeds_to_states(seeds))
    r = simulate(system, seeds)
    for reg, signed in (("x", False), ("y", False), ("vx", True), ("vy", True)):
        assert np.array_equal(q.read(st, reg, signed), r[reg]), reg


def test_circuit_size_is_independent_of_particle_count():
    """Doubling the particle count (more spawn jitter) adds no gates."""
    a = build_qps_circuit(firework(8)).dynamics
    b = build_qps_circuit(replace(firework(8), spread_bits=(3, 3))).dynamics
    assert b.size() == a.size()
    assert replace(firework(8), spread_bits=(3, 3)).num_particles == 64 * firework(8).num_particles


# ------------------------------------------------------------------ exact image
@pytest.mark.parametrize("system", [SMALL_STREAM, tiny(), replace(firework(4), screen_bits=4)])
def test_exact_image_equals_enumeration(system):
    if system.seed_bits > 22:
        pytest.skip("too many particles to enumerate")
    seeds = np.arange(system.num_particles)
    r = simulate(system, seeds)
    inten = system.intensity(r["age"])
    img = np.zeros((3, system.pixels, system.pixels))
    v = r["visible"]
    for c in range(3):
        np.add.at(img[c], (r["py"][v], r["px"][v]), inten[c][v])
    assert np.allclose(img / system.num_particles, exact_image(system), atol=1e-12)


def test_monte_carlo_splatting_converges():
    s = firework(6)
    ex = exact_image(s)
    mc = sample_image(s, 400_000, np.random.default_rng(2))
    assert np.abs(mc - ex).max() < 5 * np.sqrt(ex.max() / 400_000)


# ------------------------------------------------------------------ full A on Aer
def _regions(system):
    img = exact_image(system)
    ys, xs = np.nonzero(img[0])
    pts = [(int(x), int(y)) for x, y in zip(xs, ys)]
    return [pts[:1], pts[1:4], pts[::2]]


@pytest.mark.parametrize("channel", [None, 0, 2])
def test_objective_probability_equals_region_brightness(channel):
    s = tiny()
    q = build_qps_circuit(s)
    for region in _regions(s):
        A, obj = q.state_preparation(region, channel)
        p = AerAmplitudeSampler(A, obj).probability(0)
        assert p == pytest.approx(exact_amplitude(s, region, channel), abs=1e-10)


def test_stream_readout_uses_age_colour():
    s = replace(SMALL_STREAM, steps=3, bounce=False, margin_bits=1, kicks_x=None, kicks_y=None,
                spread_bits=(0, 0), screen_bits=2, sub_bits=1, velocities=((0, 3), (1, 2)), vel_bits=3,
                origin=(2, 0))
    q = build_qps_circuit(s)
    assert q.num_qubits + 3 <= 23
    for region in _regions(s):
        for ch in (0, 1, 2):
            A, obj = q.state_preparation(region, ch)
            p = AerAmplitudeSampler(A, obj).probability(0)
            assert p == pytest.approx(exact_amplitude(s, region, ch), abs=1e-10)


@pytest.mark.slow
def test_iqae_on_real_particle_circuits():
    """Amplitude estimation with real Q^k A circuits of a particle system."""
    s = tiny()
    q = build_qps_circuit(s)
    region = _regions(s)[1]
    A, obj = q.state_preparation(region, None)
    sampler = AerAmplitudeSampler(A, obj, seed=3)
    a = exact_amplitude(s, region, None)
    for m in (1, 3, 7):
        assert sampler.probability(m) == pytest.approx(ExactAmplitudeSampler(a).probability(m), abs=1e-9)
    r = iqae(sampler, 2e-3, alpha=1e-3, shots=32)
    assert r.interval[0] - 1e-9 <= a <= r.interval[1] + 1e-9
    assert max(r.powers) > 20


# ------------------------------------------------------------------ quantum speed-ups
def test_render_query_error_scales_like_one_over_n():
    s = firework(8)
    region = [(16, 20), (17, 20), (16, 21), (17, 21)]
    a = exact_amplitude(s, region, 0)
    rng = np.random.default_rng(4)
    cost, rmse = [], []
    for eps in (1e-3, 1e-4, 1e-5):
        runs = [iqae(ExactAmplitudeSampler(a, rng), eps, alpha=1e-3, shots=32) for _ in range(60)]
        cost.append(np.mean([r.queries for r in runs]))
        rmse.append(np.sqrt(np.mean([(r.estimate - a) ** 2 for r in runs])))
    assert np.polyfit(np.log(cost), np.log(rmse), 1)[0] < -0.8


def test_bounding_box_with_sqrt_n_evaluations():
    s = firework(4)  # 2^13 particles
    r = simulate(s, np.arange(s.num_particles))
    rng = np.random.default_rng(5)
    costs, exact = [], 0
    for _ in range(20):
        c = QueryCounter()
        i = minimum_finding(r["x"], np.ones(s.num_particles, bool), ExactGroverBackend(rng), rng, c)
        exact += r["x"][i] == r["x"].min()
        costs.append(c.total)
    assert exact >= 18
    assert np.mean(costs) < 0.15 * s.num_particles


def test_event_search_finds_rare_particles():
    s = firework(4)
    r = simulate(s, np.arange(s.num_particles))
    target = (r["px"] == r["px"].max()) & r["visible"]
    assert 0 < target.mean() < 0.02
    rng = np.random.default_rng(6)
    for _ in range(10):
        i = qsearch(target, ExactGroverBackend(rng), rng, max_rounds=200)
        assert i is not None and target[i]


def test_monte_carlo_contribution_law_matches_real_particles():
    """The multinomial MC shortcut has the statistics of really simulated particles."""
    from quantum_ray_tracing.particles import contribution_distribution, monte_carlo_estimates

    s = fountain()
    region = [(x, y) for x in range(12, 20) for y in range(0, 3)]
    vals, probs = contribution_distribution(s, region, 1)
    a = float(vals @ probs)
    assert a == pytest.approx(exact_amplitude(s, region, 1), rel=1e-12)
    rng = np.random.default_rng(9)
    seeds = s.random_seeds(400_000, rng)
    r = simulate(s, seeds)
    inreg = np.zeros(r["px"].size, bool)
    for x, y in region:
        inreg |= (r["px"] == x) & (r["py"] == y) & r["visible"]
    f = np.where(inreg, s.intensity(r["age"])[1], 0.0)
    assert f.mean() == pytest.approx(a, abs=4 * f.std() / np.sqrt(f.size))
    est = monte_carlo_estimates(vals, probs, 1000, 4000, rng)
    assert est.std() == pytest.approx(f.std() / np.sqrt(1000), rel=0.06)

"""Quantum particle systems (QPS): every particle at once, in one reversible circuit.

A classical particle system (sparks, fountains, fireworks, smoke) is a *program of
its random seed*: every particle draws its spawn offset, initial velocity, age and
per-step turbulence from random bits, then a fixed integrator moves it.  QPS takes
that literally:

    A  =  Readout  .  U_particles  .  H^{n_seed}

* ``H^{n_seed}``  - a uniform superposition over all ``N = 2^n_seed`` seeds.  Each
  basis state *is one particle*; 26 seed qubits hold 67 million particles.
* ``U_particles`` - a purely classical reversible circuit (X / multi-controlled X
  only) that emits and integrates a particle: velocity look-up table (QROM), spawn
  offset, gravity, wind, a turbulence coin per step, and semi-implicit Euler steps
  ``v += a; p += v`` in fixed-point two's-complement arithmetic.  With an age
  register in superposition it integrates a *continuous stream* of particles of
  every age at once.  Its size is that of integrating **one** particle and does not
  depend on ``N``.
* ``Readout``     - marks a set of pixels (a "camera" or "sensor" region) and
  rotates an intensity qubit by the particle's age-dependent colour, so that

      P(objective = 1) = (1/N) sum_particles intensity_c(age) * [particle in region]

  which is exactly the splatted brightness of that region in channel ``c``.

On top of ``A`` three quantum primitives give provable query speed-ups over
simulating particles one by one:

* **Render / measure** (amplitude estimation, :mod:`.amplitude_estimation`):
  region brightness with error ``O(1/M)`` after ``M`` calls of ``A``, against
  ``O(1/sqrt M)`` for classical particle sampling (Monte-Carlo splatting).
* **Bounds** (Duerr-Hoyer minimum finding, :mod:`.grover`): exact bounding box /
  extreme particle of ``N`` particles with ``O(sqrt N)`` particle evaluations
  instead of ``N``.
* **Events** (BBHT search, :mod:`.grover`): find a particle that triggers a
  collider / sub-emitter among ``N`` with ``O(sqrt(N/M))`` evaluations instead of
  ``O(N/M)``.

Everything here is checked against a classical emulator: a vectorised per-particle
integrator (used for Monte-Carlo and to verify the circuit gate-by-gate) and an
exact dynamic program over the seed distribution (the "ground truth" image).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ControlledGate
from qiskit.circuit.library import UCRYGate

__all__ = [
    "PALETTES",
    "ParticleSystem",
    "fountain",
    "firework",
    "tiny",
    "simulate",
    "exact_age_density",
    "exact_image",
    "contribution_distribution",
    "monte_carlo_estimates",
    "exact_amplitude",
    "sample_image",
    "QPSCircuit",
    "build_qps_circuit",
    "simulate_permutation",
]

# Colour ramps: (normalised age, rgb) control points, linearly interpolated.
PALETTES: dict[str, list[tuple[float, tuple[float, float, float]]]] = {
    "spark": [(0.0, (1.0, 0.97, 0.85)), (0.25, (1.0, 0.78, 0.3)), (0.6, (0.95, 0.32, 0.08)), (1.0, (0.25, 0.04, 0.02))],
    "water": [(0.0, (0.85, 0.97, 1.0)), (0.4, (0.35, 0.72, 1.0)), (1.0, (0.08, 0.22, 0.65))],
    "white": [(0.0, (1.0, 1.0, 1.0)), (1.0, (1.0, 1.0, 1.0))],
}


# --------------------------------------------------------------------------- the system
@dataclass(frozen=True)
class ParticleSystem:
    """A 2-D particle system on a fixed-point grid.

    Positions are unsigned ``pos_bits``-bit integers in units of ``2^-sub_bits``
    pixel; the visible frame is ``[0, 2^(screen_bits+sub_bits))`` on both axes and
    ``margin_bits`` extra bits hold everything that left the frame, so the image is
    ``2^screen_bits`` pixels square.  Velocities are signed ``vel_bits``-bit
    integers (units per step).

    Each step is semi-implicit Euler: ``v += (wind, -gravity) [+ kick]``,
    ``p += v`` [``+ kick``], then, with ``bounce``, a particle below the floor
    (``y < 0``, the upper half of the y range) is reflected: ``y = -y``,
    ``vy = -vy - bounce_loss``.  Turbulence ``kicks_*`` add ``kick[coin]`` with a
    fresh coin qubit per step, to the velocity or (``kick_target="position"``) to
    the position.

    ``burst``: every particle has age ``steps`` (an explosion seen ``steps`` frames
    later).  ``stream``: the age is a seed register, uniform over
    ``0 .. steps`` with ``steps + 1 = 2^age_bits`` (a continuous emitter).
    """

    name: str
    velocities: tuple[tuple[int, int], ...]  # initial (vx, vy); count must be a power of two
    origin: tuple[int, int]  # spawn position (units)
    steps: int = 8
    stream: bool = False
    screen_bits: int = 5
    sub_bits: int = 3
    vel_bits: int = 6
    spread_bits: tuple[int, int] = (0, 0)  # uniform spawn offset, in units, of 2^b
    gravity: int = 1  # subtracted from vy every step
    wind: int = 0  # added to vx every step
    kicks_x: tuple[int, int] | None = None  # turbulence: vx += kicks_x[coin] each step
    kicks_y: tuple[int, int] | None = None
    kick_target: str = "velocity"  # or "position"
    bounce: bool = False
    bounce_loss: int = 0
    margin_bits: int = 1
    palette: str = "spark"
    lifetime: float | None = None  # age that maps to the end of the palette

    def __post_init__(self):
        nv = len(self.velocities)
        if nv < 1 or nv & (nv - 1):
            raise ValueError("number of velocities must be a power of two")
        if self.stream and (self.steps + 1) & self.steps:
            raise ValueError("stream systems need steps + 1 to be a power of two")
        lo, hi = -(2 ** (self.vel_bits - 1)), 2 ** (self.vel_bits - 1) - 1
        for vx, vy in self.velocities:
            if not (lo <= vx <= hi and lo <= vy <= hi):
                raise ValueError("initial velocity out of range")
        if self.kick_target not in ("velocity", "position"):
            raise ValueError("kick_target must be 'velocity' or 'position'")
        if self.bounce and self.margin_bits < 2:
            raise ValueError("bounce needs margin_bits >= 2 (above-frame vs below-floor)")

    # ---- derived sizes
    @property
    def pos_bits(self) -> int:
        return self.screen_bits + self.sub_bits + self.margin_bits

    @property
    def world(self) -> int:
        return 2**self.pos_bits

    @property
    def visible(self) -> int:
        return 2 ** (self.screen_bits + self.sub_bits)

    @property
    def pixels(self) -> int:
        return 2**self.screen_bits

    @property
    def vidx_bits(self) -> int:
        return int(math.log2(len(self.velocities)))

    @property
    def age_bits(self) -> int:
        return int(math.log2(self.steps + 1)) if self.stream else 0

    @property
    def seed_fields(self) -> list[tuple[str, int]]:
        """Seed bit layout, least-significant first."""
        T = self.steps
        return [
            ("x_off", self.spread_bits[0]),
            ("y_off", self.spread_bits[1]),
            ("vidx", self.vidx_bits),
            ("age", self.age_bits),
            ("coin_x", T if self.kicks_x else 0),
            ("coin_y", T if self.kicks_y else 0),
        ]

    @property
    def seed_bits(self) -> int:
        return sum(b for _, b in self.seed_fields)

    @property
    def num_particles(self) -> int:
        return 2**self.seed_bits

    def ages(self) -> np.ndarray:
        return np.arange(self.steps + 1) if self.stream else np.array([self.steps])

    def intensity(self, age) -> np.ndarray:
        """RGB intensity in [0, 1] for each age; shape ``(3, len(age))``."""
        pts = PALETTES[self.palette]
        L = self.lifetime if self.lifetime is not None else max(self.steps, 1)
        t = np.clip(np.atleast_1d(np.asarray(age, float)) / L, 0, 1)
        ts = [p[0] for p in pts]
        return np.array([np.interp(t, ts, [p[1][c] for p in pts]) for c in range(3)])

    def decode_seeds(self, seeds) -> dict[str, np.ndarray]:
        """Split integer seeds (``seed_bits <= 62``) into fields; dicts pass through."""
        if isinstance(seeds, dict):
            return seeds
        if self.seed_bits > 62:
            raise ValueError("too many seed bits for integer seeds; use random_seeds()")
        seeds = np.asarray(seeds, dtype=np.int64)
        out, shift = {}, 0
        for name, b in self.seed_fields:
            out[name] = (seeds >> shift) & ((1 << b) - 1)
            shift += b
        return out

    def random_seeds(self, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """``n`` uniformly random particles (seed fields, each at most 63 bits)."""
        return {name: rng.integers(0, 1 << b, size=n, dtype=np.int64) if b else np.zeros(n, np.int64)
                for name, b in self.seed_fields}

    def with_steps(self, steps: int) -> "ParticleSystem":
        return replace(self, steps=steps)


def _ring(n: int, radius: float, jitter: float = 0.0, rng_seed: int = 0) -> tuple[tuple[int, int], ...]:
    rng = np.random.default_rng(rng_seed)
    out = []
    for k in range(n):
        a = 2 * math.pi * (k + 0.5) / n
        r = radius * (1 + jitter * (rng.random() - 0.5))
        out.append((int(round(r * math.cos(a))), int(round(r * math.sin(a)))))
    return tuple(out)


def fountain() -> ParticleSystem:
    """Continuous fountain with a bouncing floor: 16 nozzle velocities, ages 0..31.

    3 + 3 + 4 + 5 + 2*31 = 77 seed qubits: 1.5e23 particles (every drop of every age
    with every jitter history), all integrated by one circuit.
    """
    vel = tuple((vx, vy) for vy in (19, 21, 23, 25) for vx in (-5, -3, 3, 5))
    return ParticleSystem(
        name="fountain",
        velocities=vel,
        origin=(124, 0),
        steps=31,
        stream=True,
        spread_bits=(3, 3),
        gravity=2,
        kicks_x=(-2, 2),
        kicks_y=(-2, 2),
        kick_target="position",
        bounce=True,
        bounce_loss=8,
        margin_bits=2,
        palette="water",
        vel_bits=7,
        lifetime=40,
    )


def firework(steps: int = 12) -> ParticleSystem:
    """A firework burst seen ``steps`` frames after the explosion.

    32 ring velocities x a position-jitter coin per axis per step: 2^(5 + 2 steps)
    particles (2^29 = 537 million at ``steps=12``).
    """
    return ParticleSystem(
        name="firework",
        velocities=_ring(32, 8.5, jitter=0.15, rng_seed=3),
        origin=(128, 156),
        steps=steps,
        stream=False,
        gravity=1,
        kicks_x=(-2, 2),
        kicks_y=(-2, 2),
        kick_target="position",
        palette="spark",
        lifetime=16,
        vel_bits=7,
    )


def tiny(steps: int = 2) -> ParticleSystem:
    """A 4x4-pixel burst small enough to run ``Q^k A`` on a statevector simulator."""
    return ParticleSystem(
        name="tiny",
        velocities=((-1, 1), (0, 2), (1, 1), (1, 0)),
        origin=(3, 2),
        steps=steps,
        screen_bits=2,
        sub_bits=1,
        vel_bits=3,
        gravity=1,
        kicks_x=(-1, 1),
        palette="spark",
        lifetime=4,
    )


# --------------------------------------------------------------------------- classical
def _wrap_signed(v: np.ndarray, bits: int) -> np.ndarray:
    m = 1 << bits
    return ((v + m // 2) % m) - m // 2


def simulate(system: ParticleSystem, seeds: np.ndarray, check_overflow: bool = True) -> dict[str, np.ndarray]:
    """Integrate the particles with the given seeds (vectorised, exact circuit semantics)."""
    s = system
    f = s.decode_seeds(seeds)
    W = s.world
    n = next(iter(f.values())).size
    table = np.array(s.velocities, dtype=np.int64)
    x = (s.origin[0] + f["x_off"]) % W + np.zeros(n, np.int64)
    y = (s.origin[1] + f["y_off"]) % W + np.zeros(n, np.int64)
    vx = table[f["vidx"], 0].copy()
    vy = table[f["vidx"], 1].copy()
    age = f["age"] if s.stream else np.full(x.shape, s.steps)
    overflow = np.zeros(x.shape, dtype=bool)
    vel_kick = s.kick_target == "velocity"

    def kick(name, kicks, t):
        if not kicks:
            return 0
        return np.where((f[name] >> (t - 1)) & 1, kicks[1], kicks[0])

    for t in range(1, s.steps + 1):
        alive = age >= t
        kx, ky = kick("coin_x", s.kicks_x, t), kick("coin_y", s.kicks_y, t)
        nvx = vx + s.wind + (kx if vel_kick else 0)
        nvy = vy - s.gravity + (ky if vel_kick else 0)
        for v in (nvx, nvy):
            overflow |= alive & (_wrap_signed(v, s.vel_bits) != v)
        vx = np.where(alive, _wrap_signed(nvx, s.vel_bits), vx)
        vy = np.where(alive, _wrap_signed(nvy, s.vel_bits), vy)
        x = np.where(alive, (x + vx + (0 if vel_kick else kx)) % W, x)
        y = np.where(alive, (y + vy + (0 if vel_kick else ky)) % W, y)
        if s.bounce:
            hit = alive & (y >= W // 2)
            nvy = -vy - s.bounce_loss
            overflow |= hit & (_wrap_signed(nvy, s.vel_bits) != nvy)
            y = np.where(hit, (-y) % W, y)
            vy = np.where(hit, _wrap_signed(nvy, s.vel_bits), vy)
    if check_overflow and overflow.any():
        raise OverflowError(f"{s.name}: velocity leaves the {s.vel_bits}-bit range")
    vis = (x < s.visible) & (y < s.visible)
    return dict(x=x, y=y, vx=vx, vy=vy, age=age, visible=vis, px=x >> s.sub_bits, py=y >> s.sub_bits)


def _axis_dp(system: ParticleSystem, p0: np.ndarray, v0: int, accel: int, kicks, floor: bool) -> list[np.ndarray]:
    """Exact distribution of one coordinate after 0..steps steps (velocity per branch)."""
    s = system
    W = s.world
    lo, hi = -(2 ** (s.vel_bits - 1)), 2 ** (s.vel_bits - 1) - 1
    vel_kick = s.kick_target == "velocity"
    below = np.arange(W) >= W // 2
    mirror = (-np.arange(W)) % W

    def check(v):
        if not lo <= v <= hi:
            raise OverflowError(f"{s.name}: velocity leaves the {s.vel_bits}-bit range")

    def add(new, v, d):
        new[v] = new[v] + d if v in new else d

    state = {v0: p0.astype(float)}
    dists = [p0.astype(float)]
    for _ in range(s.steps):
        new: dict[int, np.ndarray] = {}
        for v, d in state.items():
            if kicks and vel_kick:
                branches = [(v + accel + k, 0.5, 0) for k in kicks]
            elif kicks:
                branches = [(v + accel, 0.5, k) for k in kicks]
            else:
                branches = [(v + accel, 1.0, 0)]
            for v2, w, dp in branches:
                check(v2)
                moved = np.roll(d, v2 + dp) * w
                if floor and s.bounce and moved[below].any():
                    hit = np.zeros(W)
                    np.add.at(hit, mirror[below], moved[below])
                    moved = np.where(below, 0.0, moved)
                    vb = -v2 - s.bounce_loss
                    check(vb)
                    add(new, vb, hit)
                add(new, v2, moved)
        state = new
        dists.append(sum(state.values()))
    return dists


def _to_pixels(system: ParticleSystem, dist: np.ndarray) -> np.ndarray:
    return dist[: system.visible].reshape(system.pixels, -1).sum(axis=1)


def exact_age_density(system: ParticleSystem) -> np.ndarray:
    """Exact fraction of all particles per (age, pixel): shape ``(n_ages, H, W)``.

    Computed with a dynamic program over the seed distribution (no enumeration of
    particles); sums to the visible fraction.
    """
    s = system
    W = s.world
    px0 = np.zeros(W)
    px0[(s.origin[0] + np.arange(2 ** s.spread_bits[0])) % W] = 2.0 ** -s.spread_bits[0]
    py0 = np.zeros(W)
    py0[(s.origin[1] + np.arange(2 ** s.spread_bits[1])) % W] = 2.0 ** -s.spread_bits[1]
    ages = s.ages()
    dens = np.zeros((ages.size, s.pixels, s.pixels))
    for vx0, vy0 in s.velocities:
        dx = _axis_dp(s, px0, vx0, s.wind, s.kicks_x, floor=False)
        dy = _axis_dp(s, py0, vy0, -s.gravity, s.kicks_y, floor=True)
        for j, T in enumerate(ages):
            dens[j] += np.outer(_to_pixels(s, dy[T]), _to_pixels(s, dx[T]))
    return dens / (ages.size * len(s.velocities))


def exact_image(system: ParticleSystem) -> np.ndarray:
    """Exact expected brightness image ``(3, H, W)`` (row 0 = bottom).

    Pixel value = (1/N) sum over all particles of intensity(age) * [particle in pixel],
    i.e. the probability that the circuit's objective qubit is 1 for that pixel.
    """
    dens = exact_age_density(system)
    inten = system.intensity(system.ages())  # (3, n_ages)
    return np.einsum("ca,ahw->chw", inten, dens)


def contribution_distribution(system: ParticleSystem, pixels, channel: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Exact law of one particle's contribution ``f = intensity(age) [in region]``.

    Returns ``(values, probabilities)``; Monte-Carlo estimates of a region's
    brightness are means of i.i.d. draws of ``f``.
    """
    dens = exact_age_density(system)
    ys = [p[1] for p in pixels]
    xs = [p[0] for p in pixels]
    p_age = dens[:, ys, xs].sum(axis=1)
    vals = np.ones(p_age.size) if channel is None else system.intensity(system.ages())[channel]
    values = np.concatenate([[0.0], vals])
    probs = np.concatenate([[max(0.0, 1 - p_age.sum())], p_age])
    return values, probs


def monte_carlo_estimates(values, probs, n: int, trials: int, rng: np.random.Generator) -> np.ndarray:
    """``trials`` independent n-particle Monte-Carlo means, drawn exactly via multinomial counts."""
    counts = rng.multinomial(n, probs / probs.sum(), size=trials)
    return counts @ values / n


def exact_amplitude(system: ParticleSystem, pixels: Sequence[tuple[int, int]], channel: int | None, image=None) -> float:
    """``P(objective = 1)`` for a pixel region; ``channel=None`` counts particles."""
    if channel is None:
        img = exact_image(replace(system, palette="white")) if image is None else image
        channel = 0
    else:
        img = exact_image(system) if image is None else image
    return float(sum(img[channel, py, px] for px, py in pixels))


def sample_image(system: ParticleSystem, n: int, rng: np.random.Generator, batch: int = 1 << 20) -> np.ndarray:
    """Classical Monte-Carlo splatting with ``n`` random particles; same units as :func:`exact_image`."""
    s = system
    img = np.zeros((3, s.pixels, s.pixels))
    done = 0
    while done < n:
        m = min(batch, n - done)
        r = simulate(s, s.random_seeds(m, rng), check_overflow=False)
        inten = s.intensity(r["age"])
        v = r["visible"]
        idx = r["py"][v] * s.pixels + r["px"][v]
        for c in range(3):
            img[c] += np.bincount(idx, weights=inten[c][v], minlength=s.pixels**2).reshape(s.pixels, s.pixels)
        done += m
    return img / n


# --------------------------------------------------------------------------- circuit
def _increment(qc: QuantumCircuit, reg: Sequence, controls: Sequence, sign: int = 1) -> None:
    """``reg += sign (mod 2^len(reg))`` when every control is |1>."""
    gates = [(list(controls) + list(reg[:b]), reg[b]) for b in range(len(reg) - 1, -1, -1)]
    if sign < 0:
        gates.reverse()
    for ctrl, tgt in gates:
        if ctrl:
            qc.mcx(ctrl, tgt)
        else:
            qc.x(tgt)


def add_constant(qc: QuantumCircuit, reg: Sequence, c: int, controls: Sequence = ()) -> None:
    """``reg += c (mod 2^k)``, controlled; uses the cheaper of +c or -(2^k - c)."""
    k = len(reg)
    c %= 2**k
    if c == 0:
        return
    sign, m = (1, c) if c <= 2 ** (k - 1) else (-1, 2**k - c)
    for j in range(k):
        if (m >> j) & 1:
            _increment(qc, reg[j:], controls, sign)


def add_register(qc: QuantumCircuit, target: Sequence, source: Sequence, controls: Sequence = ()) -> None:
    """``target += source (mod 2^len(target))`` with ``source`` in two's complement."""
    ks = len(source)
    for j in range(min(ks, len(target))):
        sign = -1 if j == ks - 1 else 1  # sign bit weighs -2^(ks-1)
        _increment(qc, target[j:], list(controls) + [source[j]], sign)


@dataclass
class QPSCircuit:
    """The particle program ``U_particles`` plus helpers to build ``A`` for a region."""

    system: ParticleSystem
    dynamics: QuantumCircuit  # U_particles (X / MCX only)
    registers: dict[str, QuantumRegister] = field(default_factory=dict)
    seed_qubits: list[int] = field(default_factory=list)

    @property
    def num_qubits(self) -> int:
        return self.dynamics.num_qubits

    def _idx(self, qubits) -> list[int]:
        return [self.dynamics.find_bit(q).index for q in qubits]

    def register_indices(self, name: str) -> list[int]:
        return self._idx(self.registers[name])

    def seeds_to_states(self, seeds) -> np.ndarray:
        """Boolean basis states ``(S, n_qubits)`` with the given seeds and all else 0."""
        f = self.system.decode_seeds(seeds)
        n = next(iter(f.values())).size
        st = np.zeros((n, self.num_qubits), dtype=bool)
        j = 0
        for name, b in self.system.seed_fields:
            for k in range(b):
                st[:, self.seed_qubits[j]] = (f[name] >> k) & 1
                j += 1
        return st

    def read(self, states: np.ndarray, name: str, signed: bool = False) -> np.ndarray:
        idx = self.register_indices(name)
        v = np.zeros(states.shape[0], dtype=np.int64)
        for j, q in enumerate(idx):
            v |= states[:, q].astype(np.int64) << j
        if signed:
            v = _wrap_signed(v, len(idx))
        return v

    def state_preparation(self, pixels: Sequence[tuple[int, int]], channel: int | None = 0) -> tuple[QuantumCircuit, int]:
        """``A = Readout . U_particles . H^seed`` and its objective qubit index.

        ``P(objective=1)`` equals :func:`exact_amplitude` for the same region/channel.
        """
        s = self.system
        hit = QuantumRegister(1, "hit")
        regs = [hit]
        if channel is not None:
            inten, out = QuantumRegister(1, "inten"), QuantumRegister(1, "out")
            regs += [inten, out]
        A = QuantumCircuit(*self.dynamics.qregs, *regs, name=f"QPS[{s.name}]")
        A.h(self.seed_qubits)
        A.compose(self.dynamics, inplace=True)
        x, y = self.registers["x"], self.registers["y"]
        hi = list(x[s.sub_bits:]) + list(y[s.sub_bits:])  # pixel bits + margin bits
        for px, py in sorted(set(pixels)):
            A.mcx(hi, hit[0], ctrl_state=px | (py << (s.screen_bits + s.margin_bits)))
        if channel is None:
            return A, A.find_bit(hit[0]).index
        angles = [2 * math.asin(math.sqrt(min(max(v, 0.0), 1.0))) for v in s.intensity(s.ages())[channel]]
        if s.stream:
            A.append(UCRYGate(angles), [inten[0], *self.registers["age"]])
        else:
            A.ry(angles[0], inten[0])
        A.ccx(hit[0], inten[0], out[0])
        return A, A.find_bit(out[0]).index


def build_qps_circuit(system: ParticleSystem) -> QPSCircuit:
    """Build the reversible particle program ``U_particles`` for ``system``."""
    s = system
    T = s.steps
    P, V = s.pos_bits, s.vel_bits
    regs = {
        "x": QuantumRegister(P, "x"),
        "y": QuantumRegister(P, "y"),
        "vx": QuantumRegister(V, "vx"),
        "vy": QuantumRegister(V, "vy"),
    }
    if s.vidx_bits:
        regs["vidx"] = QuantumRegister(s.vidx_bits, "vidx")
    if s.stream:
        regs["age"] = QuantumRegister(s.age_bits, "age")
        regs["alive"] = QuantumRegister(T, "alive")
    if s.kicks_x:
        regs["coin_x"] = QuantumRegister(T, "coin_x")
    if s.kicks_y:
        regs["coin_y"] = QuantumRegister(T, "coin_y")
    if s.bounce:
        regs["bounce"] = QuantumRegister(T, "bounce")  # which steps hit the floor (garbage)
    qc = QuantumCircuit(*regs.values(), name="U_particles")
    x, y, vx, vy = regs["x"], regs["y"], regs["vx"], regs["vy"]

    # -- emit: spawn position (seed offsets already sit in the low bits) and velocity QROM
    add_constant(qc, x, s.origin[0])
    add_constant(qc, y, s.origin[1])
    for i, (v0x, v0y) in enumerate(s.velocities):
        for reg, val in ((vx, v0x), (vy, v0y)):
            bits = [reg[b] for b in range(V) if ((val % 2**V) >> b) & 1]
            for t in bits:
                if s.vidx_bits:
                    qc.mcx(list(regs["vidx"]), t, ctrl_state=i)
                else:
                    qc.x(t)
    # -- age register -> thermometer flags alive[t-1] = [age >= t]
    if s.stream:
        age, alive = regs["age"], regs["alive"]
        for a in range(1, T + 1):
            for t in range(1, a + 1):
                qc.mcx(list(age), alive[t - 1], ctrl_state=a)
    # -- integrate: v += a (gravity, wind, turbulence); p += v; floor bounce
    vel_kick = s.kick_target == "velocity"

    def turbulence(reg, kicks, coin, ctl):
        if kicks:
            add_constant(qc, reg, kicks[1] - kicks[0], ctl + [coin])

    for t in range(1, T + 1):
        ctl = [regs["alive"][t - 1]] if s.stream else []
        kx0 = s.kicks_x[0] if s.kicks_x else 0
        ky0 = s.kicks_y[0] if s.kicks_y else 0
        cx = regs["coin_x"][t - 1] if s.kicks_x else None
        cy = regs["coin_y"][t - 1] if s.kicks_y else None
        add_constant(qc, vx, s.wind + (kx0 if vel_kick else 0), ctl)
        add_constant(qc, vy, -s.gravity + (ky0 if vel_kick else 0), ctl)
        if vel_kick:
            turbulence(vx, s.kicks_x, cx, ctl)
            turbulence(vy, s.kicks_y, cy, ctl)
        add_register(qc, x, vx, ctl)
        add_register(qc, y, vy, ctl)
        if not vel_kick:
            add_constant(qc, x, kx0, ctl)
            add_constant(qc, y, ky0, ctl)
            turbulence(x, s.kicks_x, cx, ctl)
            turbulence(y, s.kicks_y, cy, ctl)
        if s.bounce:
            # flag = [y < 0] (top bit), then y = -y and vy = -vy - loss, both as
            # "invert all bits, add 1" two's-complement negations controlled on flag.
            flag = regs["bounce"][t - 1]
            qc.mcx(ctl + [y[P - 1]], flag)
            for reg in (y, vy):
                for q in reg:
                    qc.cx(flag, q)
                _increment(qc, list(reg), [flag], 1)
            add_constant(qc, vy, -s.bounce_loss, [flag])

    seed_regs = {
        "x_off": list(x[: s.spread_bits[0]]),
        "y_off": list(y[: s.spread_bits[1]]),
        "vidx": list(regs.get("vidx", [])),
        "age": list(regs.get("age", [])),
        "coin_x": list(regs.get("coin_x", [])),
        "coin_y": list(regs.get("coin_y", [])),
    }
    seed_qubits = [qc.find_bit(q).index for name, _ in s.seed_fields for q in seed_regs[name]]
    assert len(seed_qubits) == s.seed_bits
    return QPSCircuit(system=s, dynamics=qc, registers=regs, seed_qubits=seed_qubits)


# --------------------------------------------------------------------------- gate-level check
def simulate_permutation(circuit: QuantumCircuit, states: np.ndarray) -> np.ndarray:
    """Run a classical reversible circuit (X, multi-controlled X, (C)SWAP) on basis states.

    ``states`` is a boolean array ``(S, n_qubits)``; every gate is applied as the
    permutation it implements, so this checks the *actual gates* of ``U_particles``
    on any number of particles without a statevector.
    """
    st = np.ascontiguousarray(np.asarray(states, dtype=bool).T)  # qubit-major: rows are qubits
    S = st.shape[1]
    cond = np.empty(S, dtype=bool)
    for inst in circuit.data:
        op = inst.operation
        q = [circuit.find_bit(b).index for b in inst.qubits]
        if op.name == "barrier":
            continue
        if isinstance(op, ControlledGate) and op.base_gate.name in ("x", "swap"):
            nc = op.num_ctrl_qubits
            cond.fill(True)
            for j in range(nc):
                if (op.ctrl_state >> j) & 1:
                    cond &= st[q[j]]
                else:
                    cond &= ~st[q[j]]
            if op.base_gate.name == "x":
                st[q[nc]] ^= cond
            else:
                a, b = q[nc], q[nc + 1]
                diff = (st[a] ^ st[b]) & cond
                st[a] ^= diff
                st[b] ^= diff
        elif op.name == "x":
            st[q[0]] ^= True
        elif op.name == "swap":
            st[[q[0], q[1]]] = st[[q[1], q[0]]]
        else:
            raise ValueError(f"gate {op.name!r} is not a classical reversible gate")
    return st.T

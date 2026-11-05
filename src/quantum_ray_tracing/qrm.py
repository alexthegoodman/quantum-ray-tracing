"""Quantum ray marching (Mosier, McGuire & Hachisuka, SIGGRAPH Asia 2023) in 2-D.

The paper reformulates ray marching as a discrete quantum walk on a voxel grid:

* ``Init``   - basis-encode the start voxel ``x0`` and a superposition of start
  directions ``w0``.
* ``Shift``  - move the walker one voxel along its direction (``p += dV w``).
* ``Sample`` (``M``) - amplitude-encode emission ``Le`` and reflectance ``rho`` of
  the current voxel into fresh qubits ``|Le_i>``, ``|rho_i>``.
* ``Coin``   - position-dependent coin: identity in vacuum, Lambertian scattering
  into a superposition of outgoing directions at surfaces.
* ``Path``   - with a ``|Steps>`` register in uniform superposition, multi-controlled
  X gates set the output qubit so that
  ``P(out=1) = 1/s * sum_i Le_i prod_{k<i} rho_k`` averaged over *all* paths
  (paper Eq. 15), i.e. ``F' = F / s``.

The amplitude ``F'`` is then estimated with amplitude estimation (see
:mod:`quantum_ray_tracing.amplitude_estimation`).

Reversible Lambertian coin
--------------------------
A coin that maps every incoming direction to the same outgoing distribution is
not unitary.  The paper resolves this by "shuffling the signs of the amplitudes";
we instead keep the incoming direction in a fresh *which-path* register: per
scatter event a 2-qubit register ``u`` (uniform superposition) selects one of
``{n-45deg, n, n, n+45deg}`` about the voxel normal ``n`` (a discrete cosine-like
lobe), the chosen direction is written into a fresh 3-qubit register ``c``, and a
controlled SWAP exchanges it with the direction register.  Because every path
leaves a distinct basis-state record, paths never interfere, so the objective
probability equals the exact average of ``f`` over the path distribution, which is
precisely the quantity rendered by the paper's exponential-branching emulator.

Grid and directions
-------------------
The scene is an ``8 x 8`` voxel grid (3 qubits per axis, as in the paper's 2-D
experiment).  Directions are the 8 compass directions (3 qubits), with
``index * 45deg`` and integer steps ``(dx, dy) in {-1, 0, 1}^2``.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import StatePreparation, UCRYGate

__all__ = [
    "DIRS",
    "LOBE",
    "Scene2D",
    "cornell_2d",
    "start_direction_weights",
    "enumerate_paths",
    "exact_radiance",
    "sample_paths",
    "QRMCircuit",
    "build_qrm_circuit",
]

# direction index -> integer step (dx, dy); index * 45 degrees counter-clockwise from +x
DIRS = np.array([(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)], dtype=int)
# Lambertian lobe about the normal n: u in {0,1,2,3} -> n + LOBE[u]  (weights 1/4, 1/2, 1/4)
LOBE = np.array([-1, 0, 0, 1], dtype=int)
CHANNELS = 3


@dataclass
class Scene2D:
    """Voxel scene on a ``2^bits x 2^bits`` grid with per-channel ``Le`` and ``rho``.

    ``normal[x, y]`` is ``-1`` for empty voxels and the outward normal direction
    index (0..7) for surface voxels.  Empty voxels have ``Le = 0`` and ``rho = 1``
    (vacuum does not attenuate).
    """

    bits: int
    normal: np.ndarray  # (W, H) int
    le: np.ndarray  # (C, W, H) in [0, 1]
    rho: np.ndarray  # (C, W, H) in [0, 1]

    @property
    def size(self) -> int:
        return 2**self.bits

    def surface(self) -> np.ndarray:
        return self.normal >= 0

    def cell_index(self, x: int, y: int) -> int:
        return x + self.size * y

    def flat(self, arr: np.ndarray) -> np.ndarray:
        """Flatten a (W, H) array in the circuit's basis order ``x + W*y``."""
        return np.asarray(arr).T.reshape(-1)


def cornell_2d(bits: int = 3) -> Scene2D:
    """2-D Cornell box: red left wall, green right wall, white floor/ceiling,
    an emissive ceiling patch and a white block on the floor."""
    W = 2**bits
    normal = -np.ones((W, W), dtype=int)
    rho = np.ones((CHANNELS, W, W))
    le = np.zeros((CHANNELS, W, W))
    white, red, green = (0.75, 0.75, 0.75), (0.8, 0.12, 0.1), (0.12, 0.7, 0.15)

    def wall(x, y, n, color):
        normal[x, y] = n
        rho[:, x, y] = color

    for t in range(W):
        wall(0, t, 0, red)  # left wall, normal +x (E)
        wall(W - 1, t, 4, green)  # right wall, normal -x (W)
        wall(t, 0, 2, white)  # floor, normal +y (N)
        wall(t, W - 1, 6, white)  # ceiling, normal -y (S)
    # corners: diagonal normals pointing into the room, colour of the vertical wall
    wall(0, 0, 1, red)
    wall(W - 1, 0, 3, green)
    wall(0, W - 1, 7, red)
    wall(W - 1, W - 1, 5, green)
    # area light in the middle of the ceiling: emits, absorbs everything else
    for x in range(W // 2 - 1, W // 2 + 1):
        le[:, x, W - 1] = 1.0
        rho[:, x, W - 1] = 0.0
    # a 2x2 white block standing on the floor, right of centre
    bx, by = W // 2 + 1, 1
    s = max(1, W // 8)
    for dx in range(2 * s):
        for dy in range(2 * s):
            x, y = bx + dx, by + dy
            left, top, right = dx == 0, dy == 2 * s - 1, dx == 2 * s - 1
            n = 3 if (left and top) else 1 if (right and top) else 4 if left else 0 if right else 2
            wall(x, y, n, white)
    return Scene2D(bits=bits, normal=normal, le=le, rho=rho)


# ----------------------------------------------------------------- classical emulation
def start_direction_weights(scene: Scene2D, x: int, y: int) -> np.ndarray:
    """Distribution of the initial direction: uniform in vacuum, lobe at a surface."""
    n = scene.normal[x, y]
    w = np.zeros(8)
    if n < 0:
        w[:] = 1 / 8
    else:
        for du in LOBE:
            w[(n + du) % 8] += 1 / 4
    return w


def _march(scene: Scene2D, x0: int, y0: int, d0: np.ndarray, us: np.ndarray, steps: int):
    """Vectorised walk for many paths. ``d0``: (P,), ``us``: (P, steps-1). Returns f (C, P)."""
    W = scene.size
    P = d0.size
    x = np.full(P, x0)
    y = np.full(P, y0)
    d = d0.copy()
    f = np.zeros((scene.le.shape[0], P))
    through = np.ones((scene.le.shape[0], P))
    for i in range(1, steps + 1):
        x = (x + DIRS[d, 0]) % W
        y = (y + DIRS[d, 1]) % W
        f += through * scene.le[:, x, y]
        through = through * scene.rho[:, x, y]
        if i < steps:
            n = scene.normal[x, y]
            surf = n >= 0
            d = np.where(surf, (n + LOBE[us[:, i - 1]]) % 8, d)
    return f


def enumerate_paths(scene: Scene2D, x0: int, y0: int, steps: int):
    """All branching paths (the paper's exponential 'emulated' walk) with weights."""
    w0 = start_direction_weights(scene, x0, y0)
    dirs = np.flatnonzero(w0 > 0)
    combos = list(itertools.product(range(4), repeat=steps - 1))
    us = np.array(combos, dtype=int).reshape(len(combos), steps - 1)
    d0 = np.repeat(dirs, len(combos))
    us = np.tile(us, (len(dirs), 1))
    weights = w0[d0] / len(combos)
    return _march(scene, x0, y0, d0, us, steps), weights


def exact_radiance(scene: Scene2D, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact light map ``L = Le(x0) + rho(x0) * F(x0)`` and the walk integral ``F``.

    Returns ``(L, F)`` with shapes ``(C, W, H)``.
    """
    W = scene.size
    F = np.zeros_like(scene.le)
    for x in range(W):
        for y in range(W):
            f, w = enumerate_paths(scene, x, y, steps)
            F[:, x, y] = f @ w
    return scene.le + scene.rho * F, F


def sample_paths(scene: Scene2D, x0: int, y0: int, steps: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Classical Monte-Carlo: ``n`` random walks, returns per-path ``f`` (C, n)."""
    w0 = start_direction_weights(scene, x0, y0)
    d0 = rng.choice(8, size=n, p=w0)
    us = rng.integers(0, 4, size=(n, max(steps - 1, 0)))
    return _march(scene, x0, y0, d0, us, steps)


# ----------------------------------------------------------------------- the circuit
@dataclass
class QRMCircuit:
    """The state-preparation circuit ``A`` of quantum ray marching."""

    circuit: QuantumCircuit
    objective: int
    steps: int
    scale: float  # F = scale * P(objective = 1)
    registers: dict[str, QuantumRegister] = field(default_factory=dict)


def _controlled_increment(qc: QuantumCircuit, reg, controls, ctrl_value: int, sign: int) -> None:
    """``reg += sign (mod 2^k)`` when ``controls`` are in basis state ``ctrl_value``."""
    k = len(reg)
    nc = len(controls)
    gates = []
    for b in range(k - 1, -1, -1):  # increment: flip the top bit first
        ctrl = list(controls) + [reg[j] for j in range(b)]
        state = ctrl_value | (((1 << b) - 1) << nc)
        gates.append((ctrl, reg[b], state))
    if sign < 0:
        gates.reverse()
    for ctrl, tgt, state in gates:
        qc.mcx(ctrl, tgt, ctrl_state=state)


def _shift(qc: QuantumCircuit, xr, yr, dr) -> None:
    for d, (dx, dy) in enumerate(DIRS):
        if dx:
            _controlled_increment(qc, xr, list(dr), d, int(dx))
        if dy:
            _controlled_increment(qc, yr, list(dr), d, int(dy))


def _ry_angles(p: np.ndarray) -> list[float]:
    return [float(2 * math.asin(math.sqrt(min(max(v, 0.0), 1.0)))) for v in p]


def _coin(qc: QuantumCircuit, scene: Scene2D, pos, dr, u, c, flag) -> None:
    """Position-dependent Lambertian coin via a which-path register and CSWAP."""
    nflat = scene.flat(scene.normal)
    for n in range(8):
        cells = np.flatnonzero(nflat == n)
        if cells.size == 0:
            continue
        mark = lambda: [qc.mcx(list(pos), flag, ctrl_state=int(cell)) for cell in cells]
        mark()  # flag = [pos has normal n]
        for uv in range(4):
            out = (n + LOBE[uv]) % 8
            for b in range(3):
                if (out >> b) & 1:
                    qc.mcx([flag, u[0], u[1]], c[b], ctrl_state=1 | (uv << 1))
        for b in range(3):
            qc.cswap(flag, dr[b], c[b])
        mark()  # uncompute flag (position is unchanged by the coin)


def build_qrm_circuit(
    scene: Scene2D,
    steps: int,
    channel: int = 0,
    start: tuple[int, int] | None = None,
) -> QRMCircuit:
    """Build ``A`` with ``P(objective=1) = F / steps``.

    With ``start=None`` the ``Init`` stage is omitted (positions / direction left at
    ``|0>``) so that one transpiled body can be reused for every voxel; prepend
    :func:`init_circuit` in that case.
    """
    if steps < 1:
        raise ValueError("need at least one step")
    b = scene.bits
    pos = QuantumRegister(2 * b, "pos")
    dr = QuantumRegister(3, "dir")
    le = [QuantumRegister(1, f"le{i}") for i in range(1, steps + 1)]
    rho = [QuantumRegister(1, f"rho{i}") for i in range(1, steps)]
    us = [QuantumRegister(2, f"u{i}") for i in range(1, steps)]
    cs = [QuantumRegister(3, f"c{i}") for i in range(1, steps)]
    n_step_bits = max(1, math.ceil(math.log2(steps))) if steps > 1 else 0
    regs = [pos, dr, *le, *rho, *us, *cs]
    st = QuantumRegister(n_step_bits, "steps") if n_step_bits else None
    if st is not None:
        regs.append(st)
    out = QuantumRegister(1, "out")
    regs.append(out)
    flag = QuantumRegister(1, "flag") if steps > 1 else None
    if flag is not None:
        regs.append(flag)
    qc = QuantumCircuit(*regs, name="QRM")
    xr, yr = pos[:b], pos[b:]

    if start is not None:
        qc.compose(init_circuit(scene, steps, start), qubits=list(pos) + list(dr), inplace=True)
    for u in us:
        qc.h(u)
    if st is not None:
        amps = np.zeros(2**n_step_bits)
        amps[:steps] = 1 / math.sqrt(steps)
        qc.append(StatePreparation(amps), list(st))

    le_angles = _ry_angles(scene.flat(scene.le[channel]))
    rho_angles = _ry_angles(scene.flat(scene.rho[channel]))
    for i in range(1, steps + 1):
        _shift(qc, xr, yr, dr)  # Shift S
        qc.append(UCRYGate(le_angles), [le[i - 1][0], *pos])  # Sample M: |Le_i>
        if i < steps:
            qc.append(UCRYGate(rho_angles), [rho[i - 1][0], *pos])  # Sample M: |rho_i>
            _coin(qc, scene, pos, dr, us[i - 1], cs[i - 1], flag[0])  # Coin C

    # Path gate P (paper Fig. 3): term i = Le_i * prod_{k<i} rho_k selected by |Steps>
    for i in range(1, steps + 1):
        ctrl = [le[i - 1][0]] + [rho[k][0] for k in range(i - 1)]
        state = (1 << len(ctrl)) - 1
        if st is not None:
            state |= (i - 1) << len(ctrl)
            ctrl = ctrl + list(st)
        qc.mcx(ctrl, out[0], ctrl_state=state)

    return QRMCircuit(
        circuit=qc,
        objective=qc.find_bit(out[0]).index,
        steps=steps,
        scale=float(steps),
        registers={r.name: r for r in regs},
    )


def init_circuit(scene: Scene2D, steps: int, start: tuple[int, int]) -> QuantumCircuit:
    """``Init(x0, w0)`` on the ``pos`` + ``dir`` qubits."""
    b = scene.bits
    x0, y0 = start
    qc = QuantumCircuit(2 * b + 3, name="Init")
    for j in range(b):
        if (x0 >> j) & 1:
            qc.x(j)
        if (y0 >> j) & 1:
            qc.x(b + j)
    w = start_direction_weights(scene, x0, y0)
    qc.append(StatePreparation(np.sqrt(w)), list(range(2 * b, 2 * b + 3)))
    return qc

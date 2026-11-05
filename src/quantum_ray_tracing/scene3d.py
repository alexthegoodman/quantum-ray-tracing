"""3-D scene of axis-aligned rectangles with classical and Grover-based ray tracers.

The scene mimics the "Qornell box" of Santos et al. (2022): a Cornell box built
from axis-aligned rectangles inside a 16^3 volume, a point light, two boxes, and
*dummy* primitives used to pad the primitive database to ``N = 2**n`` entries
(the paper scales from 8 to 512 primitives while keeping 14 visible).

``render_classical`` tests every primitive for every ray (``N`` intersection tests
per ray, no acceleration structure, the paper's baseline).  ``render_quantum``
finds the closest hit with Duerr-Hoyer minimum finding and resolves shadow rays
with QSearch (``QOccluded``), counting oracle calls and classical verifications.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .grover import ExactGroverBackend, GroverBackend, QueryCounter, minimum_finding, qsearch

__all__ = [
    "Rect",
    "Scene",
    "qornell_box",
    "camera_rays",
    "intersect_all",
    "RenderStats",
    "render_classical",
    "render_quantum",
    "nrmse",
    "differing_pixels",
]

EPS = 1e-4


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle: ``x[axis] == c`` with bounds on the other two axes."""

    axis: int
    c: float
    lo: tuple[float, float]  # bounds on the two remaining axes (in increasing axis order)
    hi: tuple[float, float]
    color: tuple[float, float, float] = (0.75, 0.75, 0.75)
    emission: float = 0.0
    dummy: bool = False


@dataclass
class Scene:
    rects: list[Rect]
    light: np.ndarray
    light_power: float = 1.0
    ambient: float = 0.08
    camera: np.ndarray = field(default_factory=lambda: np.array([8.0, 8.0, -15.0]))
    look_at: np.ndarray = field(default_factory=lambda: np.array([8.0, 8.0, 8.0]))
    fov_deg: float = 45.0

    @property
    def n_prims(self) -> int:
        return len(self.rects)

    @property
    def n_visible(self) -> int:
        return sum(not r.dummy for r in self.rects)

    def arrays(self):
        if not hasattr(self, "_arrays"):
            axis = np.array([r.axis for r in self.rects])
            c = np.array([r.c for r in self.rects], dtype=float)
            lo = np.array([r.lo for r in self.rects], dtype=float)
            hi = np.array([r.hi for r in self.rects], dtype=float)
            col = np.array([r.color for r in self.rects], dtype=float)
            em = np.array([r.emission for r in self.rects], dtype=float)
            self._arrays = (axis, c, lo, hi, col, em)
        return self._arrays


def _box(x0, x1, y0, y1, z0, z1, color):
    """Top and four side faces of a box (bottom face is hidden on the floor)."""
    return [
        Rect(1, y1, (x0, z0), (x1, z1), color),  # top
        Rect(2, z0, (x0, y0), (x1, y1), color),  # front
        Rect(2, z1, (x0, y0), (x1, y1), color),  # back
        Rect(0, x0, (y0, z0), (y1, z1), color),  # left
        Rect(0, x1, (y0, z0), (y1, z1), color),  # right
    ]


def qornell_box(n_prims: int = 16, seed: int = 0) -> Scene:
    """Cornell-box scene of axis-aligned rectangles padded to ``n_prims`` primitives."""
    white, red, green = (0.75, 0.75, 0.75), (0.75, 0.15, 0.12), (0.15, 0.62, 0.18)
    rects = [
        Rect(1, 0.0, (0, 0), (16, 16), white),  # floor  (y = 0)
        Rect(1, 16.0, (0, 0), (16, 16), white),  # ceiling
        Rect(2, 16.0, (0, 0), (16, 16), white),  # back wall
        Rect(0, 0.0, (0, 0), (16, 16), red),  # left wall
        Rect(0, 16.0, (0, 0), (16, 16), green),  # right wall
        Rect(1, 15.95, (6, 6), (10, 10), (1, 1, 1), emission=1.0),  # light panel
    ]
    rects += _box(3, 7, 0, 9, 8, 12, (0.7, 0.7, 0.72))  # tall box
    rects += _box(9.5, 13.5, 0, 4, 3.5, 7.5, (0.72, 0.7, 0.6))  # short box
    if n_prims < len(rects):
        raise ValueError(f"scene needs at least {len(rects)} primitives")
    rng = np.random.default_rng(seed)
    while len(rects) < n_prims:
        # Dummy primitives: real database entries that no camera/shadow ray reaches
        # (they sit behind the camera, facing away), so the image is unchanged.
        x, y = rng.uniform(0, 15, size=2)
        z = rng.uniform(-40, -20)
        rects.append(Rect(2, z, (x, y), (x + 1, y + 1), dummy=True))
    return Scene(rects=rects, light=np.array([8.0, 13.5, 2.0]), ambient=0.12)


def camera_rays(scene: Scene, width: int, height: int):
    """Pinhole camera, one ray per pixel centre. Returns origins (P,3), dirs (P,3)."""
    fwd = scene.look_at - scene.camera
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross([0.0, 1.0, 0.0], fwd)
    right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    tan = math.tan(math.radians(scene.fov_deg) / 2)
    aspect = width / height
    xs = ((np.arange(width) + 0.5) / width * 2 - 1) * tan * aspect
    ys = (1 - (np.arange(height) + 0.5) / height * 2) * tan
    X, Y = np.meshgrid(xs, ys)
    d = fwd[None, :] + X.reshape(-1, 1) * right[None, :] + Y.reshape(-1, 1) * up[None, :]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    o = np.broadcast_to(scene.camera, d.shape).copy()
    return o, d


def intersect_all(scene: Scene, origins: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Distances ``t[ray, prim]`` (``inf`` where the ray misses the primitive)."""
    axis, c, lo, hi, _, _ = scene.arrays()
    R = origins.shape[0]
    t = np.full((R, scene.n_prims), np.inf)
    others = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
    for a in range(3):
        idx = np.flatnonzero(axis == a)
        if idx.size == 0:
            continue
        u, v = others[a]
        da = dirs[:, a : a + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = (c[None, idx] - origins[:, a : a + 1]) / da
        pu = origins[:, u : u + 1] + tt * dirs[:, u : u + 1]
        pv = origins[:, v : v + 1] + tt * dirs[:, v : v + 1]
        ok = (
            (tt > EPS)
            & (pu >= lo[None, idx, 0] - 1e-9)
            & (pu <= hi[None, idx, 0] + 1e-9)
            & (pv >= lo[None, idx, 1] - 1e-9)
            & (pv <= hi[None, idx, 1] + 1e-9)
        )
        t[:, idx] = np.where(ok, tt, np.inf)
    return t


def _shade(scene: Scene, o, d, t_hit, prim, occluded) -> np.ndarray:
    """Lambert + point light + ambient. ``prim == -1`` means background."""
    axis, _, _, _, col, em = scene.arrays()
    img = np.zeros((o.shape[0], 3))
    hit = prim >= 0
    if not hit.any():
        return img
    p = o[hit] + t_hit[hit, None] * d[hit]
    pr = prim[hit]
    n = np.zeros_like(p)
    n[np.arange(p.shape[0]), axis[pr]] = 1.0
    n *= -np.sign(np.sum(n * d[hit], axis=1, keepdims=True))  # face the viewer
    L = scene.light[None, :] - p
    dist2 = np.sum(L * L, axis=1)
    L /= np.sqrt(dist2)[:, None]
    lam = np.clip(np.sum(n * L, axis=1), 0, None)
    falloff = scene.light_power * 60.0 / (dist2 + 30.0)
    direct = np.where(occluded[hit], 0.0, lam * falloff)
    c = col[pr] * (scene.ambient + direct[:, None])
    c = np.where(em[pr, None] > 0, em[pr, None] * np.ones(3), c)
    img[hit] = np.clip(c, 0, 1)
    return img


def _shadow_rays(scene: Scene, o, d, t_hit, prim):
    hit = prim >= 0
    p = o + np.where(hit, t_hit, 0)[:, None] * d
    to_l = scene.light[None, :] - p
    dist = np.linalg.norm(to_l, axis=1)
    return p, to_l / dist[:, None], dist, hit


@dataclass
class RenderStats:
    primary_rays: int = 0
    shadow_rays: int = 0
    primary: QueryCounter = field(default_factory=QueryCounter)
    shadow: QueryCounter = field(default_factory=QueryCounter)

    @property
    def rays(self) -> int:
        return self.primary_rays + self.shadow_rays

    @property
    def int_per_ray(self) -> float:
        return (self.primary.total + self.shadow.total) / max(1, self.rays)

    @property
    def int_per_primary_ray(self) -> float:
        return self.primary.total / max(1, self.primary_rays)

    @property
    def int_per_shadow_ray(self) -> float:
        return self.shadow.total / max(1, self.shadow_rays)


def render_classical(scene: Scene, width: int, height: int):
    """Exhaustive classical ray tracer: ``N`` intersection tests per ray."""
    o, d = camera_rays(scene, width, height)
    t = intersect_all(scene, o, d)
    prim = np.where(np.isfinite(t.min(axis=1)), t.argmin(axis=1), -1)
    t_hit = np.where(prim >= 0, t.min(axis=1), 0.0)
    stats = RenderStats(primary_rays=o.shape[0])
    stats.primary.classical_checks = o.shape[0] * scene.n_prims
    sp, sd, dist, need = _shadow_rays(scene, o, d, t_hit, prim)
    _, _, _, _, _, em = scene.arrays()
    need &= em[np.maximum(prim, 0)] == 0
    occ = np.zeros(o.shape[0], dtype=bool)
    if need.any():
        ts = intersect_all(scene, sp[need], sd[need])
        blocking = (ts < dist[need, None] - 1e-3) & (em[None, :] == 0)
        occ[need] = blocking.any(axis=1)
        stats.shadow_rays = int(need.sum())
        stats.shadow.classical_checks = stats.shadow_rays * scene.n_prims
    img = _shade(scene, o, d, t_hit, prim, occ)
    return img.reshape(height, width, 3), stats


def render_quantum(
    scene: Scene,
    width: int,
    height: int,
    backend: GroverBackend | None = None,
    seed: int = 0,
    c: float = 1.2,
    max_rounds: int | None = None,
    progress: bool = False,
):
    """Grover ray tracer: Duerr-Hoyer closest hit + QSearch shadow rays."""
    rng = np.random.default_rng(seed)
    backend = backend if backend is not None else ExactGroverBackend(rng)
    N = scene.n_prims
    if N & (N - 1):
        raise ValueError("number of primitives must be a power of two")
    o, d = camera_rays(scene, width, height)
    t_all = intersect_all(scene, o, d)  # the classical predicate table defining the oracle
    stats = RenderStats(primary_rays=o.shape[0])
    prim = np.full(o.shape[0], -1)
    t_hit = np.zeros(o.shape[0])
    for i in range(o.shape[0]):
        ti = t_all[i]
        j = minimum_finding(ti, np.isfinite(ti), backend, rng, stats.primary, c=c, max_rounds=max_rounds)
        if j is not None:
            prim[i], t_hit[i] = j, ti[j]
        if progress and i % 1000 == 0:
            print(f"  primary {i}/{o.shape[0]}", flush=True)
    sp, sd, dist, need = _shadow_rays(scene, o, d, t_hit, prim)
    _, _, _, _, _, em = scene.arrays()
    need &= em[np.maximum(prim, 0)] == 0
    occ = np.zeros(o.shape[0], dtype=bool)
    idx = np.flatnonzero(need)
    if idx.size:
        ts = intersect_all(scene, sp[idx], sd[idx])
        blocking = (ts < dist[idx, None] - 1e-3) & (em[None, :] == 0)
        stats.shadow_rays = idx.size
        for k, i in enumerate(idx):
            occ[i] = qsearch(blocking[k], backend, rng, stats.shadow, c=c, max_rounds=max_rounds) is not None
    img = _shade(scene, o, d, t_hit, prim, occ)
    return img.reshape(height, width, 3), stats


def nrmse(img: np.ndarray, ref: np.ndarray) -> float:
    """Root-mean-square error normalised by the reference's value range."""
    rng = float(ref.max() - ref.min()) or 1.0
    return float(np.sqrt(np.mean((img - ref) ** 2)) / rng)


def differing_pixels(img: np.ndarray, ref: np.ndarray, tol: float = 1e-6) -> int:
    return int(np.any(np.abs(img - ref) > tol, axis=-1).sum())

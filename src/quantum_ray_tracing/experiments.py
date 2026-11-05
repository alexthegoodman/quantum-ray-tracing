"""Reproduce all figures and numbers: ``uv run qrt-experiments [--quick]``.

Outputs go to ``results/`` (PNG figures + ``results.json``).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from .amplitude_estimation import AerAmplitudeSampler, ExactAmplitudeSampler, iqae
from .grover import AerGroverBackend, ExactGroverBackend
from .qrm import build_qrm_circuit, cornell_2d, enumerate_paths, exact_radiance
from .scene3d import differing_pixels, nrmse, qornell_box, render_classical, render_quantum

# Reference palette (validated categorical slots 1-2, neutral ink for references).
QUANTUM = "#2a78d6"
CLASSICAL = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#9a9893"
SURFACE = "#fcfcfb"

# Amplitude-estimation settings used throughout (IQAE, Grinko et al. 2021).
ALPHA = 1e-3
SHOTS = 32

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.grid": True,
        "grid.color": "#e6e5e1",
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
    }
)


def _loglog_slope(x, y) -> float:
    x, y = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    return float(np.polyfit(x, y, 1)[0])


# ----------------------------------------------------------------------------- Santos
def exp_grover_scaling(out: Path, quick: bool) -> dict:
    """Intersection tests per ray vs. number of primitives (Santos et al. Table 2)."""
    Ns = [16, 32, 64, 128, 256] if quick else [16, 32, 64, 128, 256, 512]
    res = 48 if quick else 96
    rows = []
    for N in Ns:
        scene = qornell_box(N)
        ref, cst = render_classical(scene, res, res)
        img, qst = render_quantum(scene, res, res, seed=N)
        rows.append(
            dict(
                N=N,
                classical_int_per_ray=cst.int_per_ray,
                quantum_int_per_ray=qst.int_per_ray,
                quantum_int_per_primary_ray=qst.int_per_primary_ray,
                quantum_int_per_shadow_ray=qst.int_per_shadow_ray,
                oracle_calls_per_ray=(qst.primary.oracle_calls + qst.shadow.oracle_calls) / qst.rays,
                nrmse=nrmse(img, ref),
                differing_pixels=differing_pixels(img, ref),
                pixels=res * res,
            )
        )
        print(f"  N={N:4d}  classical {cst.int_per_ray:6.1f}  quantum {qst.int_per_ray:6.1f}  NRMSE {rows[-1]['nrmse']:.4f}")
    N = np.array([r["N"] for r in rows])
    q = np.array([r["quantum_int_per_ray"] for r in rows])
    slope = _loglog_slope(N, q)
    a = float(np.exp(np.mean(np.log(q) - 0.5 * np.log(N))))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(N, N, color=CLASSICAL, lw=2, marker="o", ms=8, label="classical (test every primitive)")
    ax.plot(N, q, color=QUANTUM, lw=2, marker="o", ms=8, label="Grover / Duerr-Hoyer (measured)")
    ax.plot(N, a * np.sqrt(N), color=MUTED, lw=1.2, ls="--", label=f"{a:.1f}·√N")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("primitives in the scene, N")
    ax.set_ylabel("intersection evaluations per ray")
    ax.set_title(f"Ray-scene intersection cost (fitted slope {slope:.2f}; √N = 0.5)", loc="left")
    ax.text(N[-1], N[-1], "  classical", color=INK2, va="center", ha="left")
    ax.text(N[-1], q[-1], "  quantum", color=INK2, va="center", ha="left")
    ax.set_xlim(N[0] / 1.3, N[-1] * 2.2)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig1_grover_scaling.png", dpi=160)
    plt.close(fig)
    crossover = next((int(r["N"]) for r in rows if r["quantum_int_per_ray"] < r["classical_int_per_ray"]), None)
    return dict(rows=rows, loglog_slope=slope, sqrt_coefficient=a, first_N_quantum_cheaper=crossover)


def exp_qornell_render(out: Path, quick: bool) -> dict:
    """Images: classical reference, Grover (ideal sampler), Grover executed on Aer."""
    res = 64 if quick else 128
    scene = qornell_box(64)
    ref, cst = render_classical(scene, res, res)
    img, qst = render_quantum(scene, res, res, seed=7)
    # Real circuits on Qiskit Aer (one shot per Grover circuit) - smaller image.
    ares = 32 if quick else 64
    aref, _ = render_classical(scene, ares, ares)
    backend = AerGroverBackend(seed=11)
    t0 = time.time()
    aimg, ast = render_quantum(scene, ares, ares, backend=backend, seed=11)
    aer_time = time.time() - t0

    fig, axs = plt.subplots(1, 4, figsize=(13, 4.0))
    panels = [
        (ref, f"Classical reference\nN=64, {cst.int_per_ray:.0f} tests/ray"),
        (img, f"Quantum (ideal Grover)\nN=64, {qst.int_per_ray:.1f} evals/ray, NRMSE {nrmse(img, ref):.3f}"),
        (np.clip(np.abs(img - ref).max(axis=-1) * 4, 0, 1), f"|quantum − classical| ×4\n{differing_pixels(img, ref)} of {res*res} pixels differ"),
        (aimg, f"Grover circuits run on Qiskit Aer\nN=64, {ares}×{ares}, {backend.circuits_run} circuits, NRMSE {nrmse(aimg, aref):.3f}"),
    ]
    for ax, (im, title) in zip(axs, panels):
        ax.imshow(im, cmap="gray" if im.ndim == 2 else None, interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9, fontweight="normal")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "fig2_qornell_box.png", dpi=160)
    plt.close(fig)
    plt.imsave(out / "qornell_quantum.png", np.clip(img, 0, 1))
    return dict(
        resolution=res,
        N=64,
        classical_int_per_ray=cst.int_per_ray,
        quantum_int_per_ray=qst.int_per_ray,
        nrmse=nrmse(img, ref),
        differing_pixels=differing_pixels(img, ref),
        aer=dict(
            resolution=ares,
            N=64,
            circuits=backend.circuits_run,
            seconds=aer_time,
            nrmse=nrmse(aimg, aref),
            differing_pixels=differing_pixels(aimg, aref),
            int_per_ray=ast.int_per_ray,
        ),
    )


# ------------------------------------------------------------------------------ Mosier
def circuit_light_map(scene, steps: int) -> np.ndarray:
    """Light map from the *actual* QRM circuits (Aer statevector / MPS), every voxel."""
    method = "statevector" if steps <= 2 else "matrix_product_state"
    sim = AerSimulator(method=method)
    W = scene.size
    F = np.zeros_like(scene.le)
    for ch in range(scene.le.shape[0]):
        for x in range(W):
            for y in range(W):
                q = build_qrm_circuit(scene, steps, ch, (x, y))
                qc = q.circuit.copy()
                qc.save_probabilities([q.objective])
                p = sim.run(transpile(qc, sim, optimization_level=1)).result().data()["probabilities"][1]
                F[ch, x, y] = q.scale * p
    return scene.le + scene.rho * F, F


def iqae_light_map(scene, steps: int, F_exact: np.ndarray, eps: float, rng):
    """Quantum light map: IQAE per voxel and channel on the exact amplitude law.

    Returns ``(L, queries)`` where ``queries`` holds the calls of ``A`` per entry.
    """
    F = np.zeros_like(F_exact)
    Q = np.zeros(F_exact.shape, dtype=int)
    for idx in np.ndindex(F.shape):
        r = iqae(ExactAmplitudeSampler(F_exact[idx] / steps, rng), eps, alpha=ALPHA, shots=SHOTS)
        F[idx], Q[idx] = steps * r.estimate, r.queries
    return scene.le + scene.rho * F, Q


def mc_light_map(scene, steps: int, budget: np.ndarray, rng) -> np.ndarray:
    """Classical Monte Carlo with ``budget[c, x, y]`` sampled paths per entry."""
    W = scene.size
    F = np.zeros_like(scene.le)
    for x in range(W):
        for y in range(W):
            f, w = enumerate_paths(scene, x, y, steps)
            for ch in range(F.shape[0]):
                pick = rng.choice(w.size, size=int(budget[ch, x, y]), p=w)
                F[ch, x, y] = f[ch, pick].mean()
    return scene.le + scene.rho * F


def _tonemap(L: np.ndarray) -> np.ndarray:
    """(C, W, H) radiance -> displayable (H, W, C) image, y up."""
    img = np.transpose(L, (2, 1, 0))[::-1]
    return np.clip(img / (1 + img), 0, 1) ** (1 / 2.2) * 1.6


def _show_map(ax, L, title, smooth=False):
    ax.imshow(np.clip(_tonemap(L), 0, 1), interpolation="bilinear" if smooth else "nearest")
    ax.set_title(title, fontsize=9, fontweight="normal")
    ax.axis("off")


def _rmse(a, b) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def exp_qrm_light_maps(out: Path, quick: bool) -> dict:
    scene = cornell_2d()
    results = {}
    t0 = time.time()
    L2_circ, _ = circuit_light_map(scene, 2)
    L2_ex, _ = exact_radiance(scene, 2)
    results["s2_circuit_seconds"] = time.time() - t0
    results["s2_circuit_vs_exact_max_abs"] = float(np.abs(L2_circ - L2_ex).max())
    t0 = time.time()
    L3_circ, _ = circuit_light_map(scene, 3)
    L3_ex, F3 = exact_radiance(scene, 3)
    results["s3_circuit_seconds"] = time.time() - t0
    results["s3_circuit_vs_exact_max_abs"] = float(np.abs(L3_circ - L3_ex).max())
    print(f"  circuit light maps: |s2 - exact| = {results['s2_circuit_vs_exact_max_abs']:.2e}, "
          f"|s3 - exact| = {results['s3_circuit_vs_exact_max_abs']:.2e}")

    rng = np.random.default_rng(5)
    eps = 1e-3
    results["iqae_epsilon"] = eps
    # quantum estimates use the circuit's own amplitudes (identical to exact up to 1e-11)
    F3_circ = (L3_circ - scene.le) / np.where(scene.rho > 0, scene.rho, 1)
    F3_circ = np.where(scene.rho > 0, F3_circ, F3)
    L3_q, Q3 = iqae_light_map(scene, 3, F3_circ, eps, rng)
    L3_mc = mc_light_map(scene, 3, Q3, rng)  # same number of evaluations per voxel
    results["s3_mean_calls_per_voxel"] = float(Q3.mean())
    results["s3_rmse_quantum"] = _rmse(L3_q, L3_ex)
    results["s3_rmse_mc"] = _rmse(L3_mc, L3_ex)

    # Longer walks: the paper's "emulated" mode (exponential classical branching).
    long_steps = 6 if quick else 8
    Ll_ex, Fl = exact_radiance(scene, long_steps)
    Ll_q, Ql = iqae_light_map(scene, long_steps, Fl, eps, rng)
    Ll_mc = mc_light_map(scene, long_steps, Ql, rng)
    results["long_steps"] = long_steps
    results["long_mean_calls_per_voxel"] = float(Ql.mean())
    results["long_rmse_quantum"] = _rmse(Ll_q, Ll_ex)
    results["long_rmse_mc"] = _rmse(Ll_mc, Ll_ex)
    print(f"  equal budget (~{Q3.mean():.0f} calls/voxel): RMSE quantum {results['s3_rmse_quantum']:.2e} "
          f"vs MC {results['s3_rmse_mc']:.2e} (3 steps); {results['long_rmse_quantum']:.2e} vs "
          f"{results['long_rmse_mc']:.2e} ({long_steps} steps)")

    n3, nl = int(round(Q3.mean(), -2)), int(round(Ql.mean(), -2))
    fig, axs = plt.subplots(2, 5, figsize=(14, 6.9))
    _show_map(axs[0, 0], L3_ex, "exact (all 128 paths/voxel)\n3 steps")
    _show_map(axs[0, 1], L3_circ, "QRM circuit on Aer (28 qubits)\n3 steps")
    _show_map(axs[0, 2], L3_q, f"quantum AE, ~{n3} calls/voxel\nRMSE {results['s3_rmse_quantum']:.1e}")
    _show_map(axs[0, 3], L3_mc, f"classical MC, same budget\nRMSE {results['s3_rmse_mc']:.1e}")
    _show_map(axs[0, 4], L3_circ, "circuit light map,\nbilinear upsampling", smooth=True)
    _show_map(axs[1, 0], Ll_ex, f"exact, {long_steps} steps\n(emulated branching)")
    _show_map(axs[1, 1], L2_circ, "QRM circuit on Aer (20 qubits)\n2 steps")
    _show_map(axs[1, 2], Ll_q, f"quantum AE, ~{nl} calls/voxel\nRMSE {results['long_rmse_quantum']:.1e}")
    _show_map(axs[1, 3], Ll_mc, f"classical MC, same budget\nRMSE {results['long_rmse_mc']:.1e}")
    _show_map(axs[1, 4], Ll_q, f"quantum AE, {long_steps} steps,\nbilinear upsampling", smooth=True)
    fig.suptitle("Quantum ray marching: 8×8 light maps of a 2-D Cornell box", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(h_pad=2.0)
    fig.savefig(out / "fig3_qrm_light_maps.png", dpi=160)
    plt.close(fig)
    # Error maps on one shared sequential scale (max over channels, per voxel).
    errs = [
        (np.abs(L3_q - L3_ex).max(axis=0), f"quantum AE, 3 steps\nRMSE {results['s3_rmse_quantum']:.1e}"),
        (np.abs(L3_mc - L3_ex).max(axis=0), f"classical MC, 3 steps\nRMSE {results['s3_rmse_mc']:.1e}"),
        (np.abs(Ll_q - Ll_ex).max(axis=0), f"quantum AE, {long_steps} steps\nRMSE {results['long_rmse_quantum']:.1e}"),
        (np.abs(Ll_mc - Ll_ex).max(axis=0), f"classical MC, {long_steps} steps\nRMSE {results['long_rmse_mc']:.1e}"),
    ]
    vmax = max(e.max() for e, _ in errs)
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.6))
    for ax, (e, title) in zip(axs, errs):
        im = ax.imshow(e.T[::-1], cmap="Blues", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=9, fontweight="normal")
        ax.axis("off")
    cb = fig.colorbar(im, ax=axs, fraction=0.02, pad=0.02)
    cb.set_label("|error| per voxel (same budget)", color=INK2)
    fig.suptitle("Absolute error of the light maps at an equal evaluation budget", x=0.01, ha="left", fontweight="bold")
    fig.savefig(out / "fig5_error_maps.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    big = _tonemap(Ll_q)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(np.clip(big, 0, 1), interpolation="bilinear")
    ax.axis("off")
    fig.savefig(out / "qrm_quantum_light_map.png", dpi=128, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return results


def exp_convergence(out: Path, quick: bool) -> dict:
    """RMSE vs. ray-marching evaluations: quantum AE O(1/N) vs Monte Carlo O(1/sqrt N)."""
    scene = cornell_2d()
    steps = 3
    voxels = [(2, 5), (4, 4), (5, 6), (1, 6), (6, 5), (3, 2)]
    trials = 40 if quick else 100
    epsilons = [3e-2, 1e-2, 3e-3, 1e-3, 3e-4] + ([] if quick else [1e-4])
    rng = np.random.default_rng(1)
    cases = []
    for (x, y) in voxels:
        f, w = enumerate_paths(scene, x, y, steps)
        cases.append((f[0], w, float(f[0] @ w)))
    q_cost, q_rmse, c_rmse = [], [], []
    for eps in epsilons:
        err_q, cost = [], []
        for f, w, F in cases:
            for _ in range(trials):
                r = iqae(ExactAmplitudeSampler(F / steps, rng), eps, alpha=ALPHA, shots=SHOTS)
                err_q.append(steps * r.estimate - F)
                cost.append(r.queries)
        n = int(np.mean(cost))
        err_c = []
        for f, w, F in cases:
            for _ in range(trials):
                pick = rng.choice(w.size, size=n, p=w)
                err_c.append(f[pick].mean() - F)
        q_cost.append(n)
        q_rmse.append(_rmse(np.array(err_q), 0))
        c_rmse.append(_rmse(np.array(err_c), 0))
        print(f"  eps={eps:.0e}  N={n:7d}  quantum {q_rmse[-1]:.2e}  MC {c_rmse[-1]:.2e}")

    # The same algorithm on real Grover circuits Q^k A executed by Qiskit Aer.
    # 1-step walk (11 qubits) so that many trials at deep Grover powers are affordable.
    aer_eps = [3e-2, 1e-2, 3e-3, 1e-3]
    aer_trials = 5 if quick else 10
    aer_cases = []
    for (x, y) in [(3, 6), (2, 6)]:
        q = build_qrm_circuit(scene, 1, 0, (x, y))
        f, w = enumerate_paths(scene, x, y, 1)
        aer_cases.append((AerAmplitudeSampler(q.circuit, q.objective, seed=x * 10 + y), float(f[0] @ w)))
    a_cost, a_rmse, a_maxk = [], [], []
    t0 = time.time()
    for eps in aer_eps:
        errs, cost, maxk = [], [], 0
        for sampler, F in aer_cases:
            for _ in range(aer_trials):
                r = iqae(sampler, eps, alpha=ALPHA, shots=SHOTS)
                errs.append(r.estimate - F)
                cost.append(r.queries)
                maxk = max(maxk, max(r.powers))
        a_cost.append(float(np.mean(cost)))
        a_rmse.append(_rmse(np.array(errs), 0))
        a_maxk.append(maxk)
        print(f"  Aer eps={eps:.0e}  N={a_cost[-1]:7.0f}  quantum {a_rmse[-1]:.2e}  max Grover power {maxk}  ({time.time() - t0:.0f}s)")

    # One run on the 20-qubit 2-step walk with real Grover circuits.
    q2 = build_qrm_circuit(scene, 2, 0, (3, 5))
    f2, w2 = enumerate_paths(scene, 3, 5, 2)
    t0 = time.time()
    r2 = iqae(AerAmplitudeSampler(q2.circuit, q2.objective, seed=7), 1e-2, alpha=ALPHA, shots=SHOTS)
    two_step = dict(qubits=q2.circuit.num_qubits, exact_F=float(f2[0] @ w2), estimate_F=2 * r2.estimate,
                    interval_F=[2 * r2.interval[0], 2 * r2.interval[1]], queries=r2.queries,
                    max_grover_power=max(r2.powers), seconds=time.time() - t0)
    print(f"  Aer 2-step walk: F = {two_step['estimate_F']:.4f} (exact {two_step['exact_F']:.4f}), "
          f"{r2.queries} calls, {two_step['seconds']:.0f}s")

    sq = _loglog_slope(q_cost, q_rmse)
    sc = _loglog_slope(q_cost, c_rmse)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.plot(q_cost, c_rmse, color=CLASSICAL, lw=2, marker="o", ms=8, label=f"classical Monte Carlo (slope {sc:.2f})")
    ax.plot(q_cost, q_rmse, color=QUANTUM, lw=2, marker="o", ms=8, label=f"quantum amplitude estimation (slope {sq:.2f})")
    ax.plot(a_cost, a_rmse, color=QUANTUM, lw=0, marker="D", ms=8, mfc=SURFACE, mew=2,
            label="quantum AE, real Grover circuits on Aer (1-step walk)")
    x = np.array([min(q_cost), max(q_cost)], float)
    ax.plot(x, c_rmse[0] * (x / x[0]) ** -0.5, color=MUTED, lw=1, ls="--")
    ax.plot(x, q_rmse[0] * (x / x[0]) ** -1.0, color=MUTED, lw=1, ls=":")
    ax.text(x[1] * 1.2, c_rmse[0] * (x[1] / x[0]) ** -0.5, "1/√N", color=INK2, va="center")
    ax.text(x[1] * 1.2, q_rmse[0] * (x[1] / x[0]) ** -1.0, "1/N", color=INK2, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x[0] / 1.5, x[1] * 2.5)
    ax.set_xlabel("ray-marching circuit evaluations per voxel, N")
    ax.set_ylabel("RMSE of voxel radiance")
    ax.set_title("Error convergence: quantum ray marching vs Monte Carlo", loc="left")
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig4_convergence.png", dpi=160)
    plt.close(fig)
    # How many more paths MC needs to match the quantum error at the largest budget
    mc_needed = q_cost[0] * (c_rmse[0] / q_rmse[-1]) ** 2  # MC follows 1/sqrt(N) exactly
    return dict(
        voxels=voxels, steps=steps, trials_per_voxel=trials, shots=SHOTS, alpha=ALPHA, epsilons=epsilons,
        cost=q_cost, quantum_rmse=q_rmse, mc_rmse=c_rmse,
        quantum_slope=sq, mc_slope=sc,
        mc_paths_to_match_best_quantum=float(mc_needed),
        cost_ratio_at_best_quantum_error=float(mc_needed / q_cost[-1]),
        aer=dict(epsilons=aer_eps, cost=a_cost, rmse=a_rmse, max_grover_power=a_maxk,
                 trials=aer_trials * len(aer_cases)),
        aer_two_step=two_step,
    )


def circuit_stats() -> dict:
    scene = cornell_2d()
    sim = AerSimulator()
    stats = {}
    for s in (1, 2, 3, 4):
        q = build_qrm_circuit(scene, s, 0, (3, 4))
        t = transpile(q.circuit, basis_gates=["cx", "u"], optimization_level=1)
        ops = t.count_ops()
        stats[s] = dict(qubits=q.circuit.num_qubits, depth=t.depth(), cx=int(ops.get("cx", 0)))
    return stats


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="smaller sizes / fewer trials")
    ap.add_argument("--out", default="results")
    ap.add_argument("--only", nargs="*", default=None, help="subset: scaling render qrm convergence stats")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "results.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    stages = {
        "scaling": lambda: exp_grover_scaling(out, args.quick),
        "render": lambda: exp_qornell_render(out, args.quick),
        "qrm": lambda: exp_qrm_light_maps(out, args.quick),
        "convergence": lambda: exp_convergence(out, args.quick),
        "stats": circuit_stats,
    }
    for name, fn in stages.items():
        if args.only and name not in args.only:
            continue
        print(f"[{name}]", flush=True)
        t0 = time.time()
        results[name] = fn()
        results[name + "_seconds"] = time.time() - t0
        path.write_text(json.dumps(results, indent=2, default=float))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

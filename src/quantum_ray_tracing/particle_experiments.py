"""Quantum particle systems: figures and numbers.  ``uv run qps-experiments [--quick]``.

Outputs go to ``results/`` (``fig6``-``fig9``, ``qps_firework.gif`` and
``results/qps_results.json``).
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
from matplotlib.animation import FuncAnimation, PillowWriter

from .amplitude_estimation import AerAmplitudeSampler, ExactAmplitudeSampler, iqae
from .experiments import CLASSICAL, INK, INK2, MUTED, QUANTUM, SURFACE, _loglog_slope  # noqa: F401 (style)
from .grover import ExactGroverBackend, QueryCounter, minimum_finding, qsearch
from .particles import (
    build_qps_circuit,
    contribution_distribution,
    exact_amplitude,
    exact_image,
    firework,
    fountain,
    monte_carlo_estimates,
    simulate,
    simulate_permutation,
    tiny,
)

ALPHA = 1e-3
SHOTS = 32
NIGHT = "#07080c"


# ----------------------------------------------------------------------------- helpers
def tonemap(img: np.ndarray, white: float | None = None) -> np.ndarray:
    """(3, H, W) linear brightness -> (H, W, 3) display RGB (row 0 at the bottom)."""
    x = np.moveaxis(img, 0, -1)
    if white is None:
        white = max(float(np.percentile(x.max(axis=2), 99.7)), 1e-15)
    return np.clip(x / white, 0, 1) ** (1 / 2.2)


def show(ax, img, title=None, white=None, smooth=True):
    ax.imshow(tonemap(img, white), origin="lower", interpolation="bilinear" if smooth else "nearest")
    ax.set_facecolor(NIGHT)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, fontsize=9, fontweight="normal")


def circuit_sampled_image(system, n: int, rng) -> tuple[np.ndarray, int]:
    """Splat ``n`` particles obtained by running the *gates* of U_particles on random seeds.

    Also returns the number of particles whose circuit result differs from the
    classical integrator (expected: 0).
    """
    q = build_qps_circuit(system)
    seeds = system.random_seeds(n, rng)
    st = simulate_permutation(q.dynamics, q.seeds_to_states(seeds))
    x, y = q.read(st, "x"), q.read(st, "y")
    ref = simulate(system, seeds)
    mismatches = int(np.count_nonzero((x != ref["x"]) | (y != ref["y"])))
    age = q.read(st, "age") if system.stream else np.full(n, system.steps)
    vis = (x < system.visible) & (y < system.visible)
    px, py = x[vis] >> system.sub_bits, y[vis] >> system.sub_bits
    inten = system.intensity(age[vis])
    img = np.zeros((3, system.pixels, system.pixels))
    for c in range(3):
        np.add.at(img[c], (py, px), inten[c])
    return img / n, mismatches


def tile(x0, y0, w, h):
    return [(x, y) for x in range(x0, x0 + w) for y in range(y0, y0 + h)]


# ----------------------------------------------------------------------------- figures
def exp_frames(out: Path, quick: bool) -> dict:
    """Fig. 6 + GIF: the circuit *is* the particle system."""
    rng = np.random.default_rng(0)
    n = 30_000 if quick else 150_000
    times = [2, 5, 8, 11, 14]
    systems = [fountain()] + [firework(t) for t in times]
    exact = [exact_image(s) for s in systems]
    sampled, mism = [], 0
    t0 = time.time()
    for s in systems:
        img, m = circuit_sampled_image(s, n, rng)
        sampled.append(img)
        mism += m
    res = dict(
        particles_per_frame_gate_level=n,
        gate_level_mismatches=mism,
        gate_level_seconds=time.time() - t0,
        fountain_particles=fountain().num_particles,
        firework_particles={t: firework(t).num_particles for t in times},
    )
    print(f"  gate-level frames: {mism} mismatches vs integrator over {n * len(systems):,} particles")

    def white_of(img):  # per-frame exposure; the palette still carries the cooling colour
        return float(np.percentile(np.moveaxis(img, 0, -1).max(2), 99.7))

    with plt.rc_context({"axes.grid": False}):
        fig, axs = plt.subplots(2, 6, figsize=(15, 6.3))
        for j, s in enumerate(systems):
            white = white_of(exact[j])
            label = "fountain (ages 0-31)" if j == 0 else f"firework, t = {s.steps}"
            e = int(math.floor(math.log10(s.num_particles)))
            show(axs[0, j], exact[j], f"{label}\nexact, all {s.num_particles / 10**e:.1f}e{e} particles", white)
            show(axs[1, j], sampled[j], f"{n:,} particles run through\nthe gates of U_particles", white)
        fig.suptitle(
            "Quantum particle systems: the reversible circuit reproduces the particle integrator exactly",
            x=0.01, ha="left", fontweight="bold",
        )
        fig.tight_layout(h_pad=2.5)
        fig.savefig(out / "fig6_qps_frames.png", dpi=150)
        plt.close(fig)

        # animation of the burst (exact images; every frame is one circuit with steps = t)
        T = list(range(0, 17))
        frames = [exact_image(firework(t)) for t in T]
        fnt = exact[0]
        fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.4), facecolor=NIGHT)
        fig.subplots_adjust(0.01, 0.01, 0.99, 0.88, wspace=0.04)
        im0 = axs[0].imshow(tonemap(frames[0], white_of(frames[0])), origin="lower", interpolation="bilinear")
        axs[1].imshow(tonemap(fnt), origin="lower", interpolation="bilinear")
        for ax, t in zip(axs, ["firework  (2^(5+2t) particles)", "fountain  (2^77 particles)"]):
            ax.axis("off")
            ax.set_title(t, color="#d8d8d8", fontsize=9)

        def upd(i):
            im0.set_data(tonemap(frames[i], white_of(frames[i])))
            return (im0,)

        anim = FuncAnimation(fig, upd, frames=len(frames), blit=True)
        anim.save(out / "qps_firework.gif", writer=PillowWriter(fps=6), dpi=90, savefig_kwargs={"facecolor": NIGHT})
        plt.close(fig)
    return res


def exp_convergence(out: Path, quick: bool) -> dict:
    """Fig. 7: region brightness, quantum AE O(1/M) vs particle sampling O(1/sqrt M)."""
    trials = 40 if quick else 100
    epsilons = [3e-2, 1e-2, 3e-3, 1e-3, 3e-4] + ([] if quick else [1e-4, 3e-5])
    queries = [
        ("fountain: splash on the basin floor (green)", fountain(), tile(10, 0, 12, 2), 1),
        ("fountain: spray above head height (count)", fountain(), tile(0, 14, 32, 18), None),
        ("firework t=8: light on a 4x4 sensor (red)", firework(8), tile(18, 20, 4, 4), 0),
        ("firework t=12: sparks below the horizon (count)", firework(12), tile(0, 0, 32, 10), None),
    ]
    rng = np.random.default_rng(1)
    rows = []
    q_cost = np.zeros(len(epsilons))
    q_se = np.zeros(len(epsilons))
    c_se = np.zeros(len(epsilons))
    for name, s, region, ch in queries:
        vals, probs = contribution_distribution(s, region, ch)
        a = float(vals @ probs)
        rows.append(dict(query=name, amplitude=a, particles=float(s.num_particles)))
        for k, eps in enumerate(epsilons):
            runs = [iqae(ExactAmplitudeSampler(a, rng), eps, alpha=ALPHA, shots=SHOTS) for _ in range(trials)]
            n = int(np.mean([r.queries for r in runs]))
            q_cost[k] += n / len(queries)
            q_se[k] += np.mean([(r.estimate - a) ** 2 for r in runs]) / len(queries)
            c_se[k] += np.mean((monte_carlo_estimates(vals, probs, n, trials, rng) - a) ** 2) / len(queries)
    q_rmse, c_rmse = np.sqrt(q_se), np.sqrt(c_se)
    sq, sc = _loglog_slope(q_cost, q_rmse), _loglog_slope(q_cost, c_rmse)
    # Monte-Carlo cost to match the most precise quantum point (MC error = sigma / sqrt n)
    c_fit = np.exp(np.mean(np.log(c_rmse) + 0.5 * np.log(q_cost)))
    n_match = (c_fit / q_rmse[-1]) ** 2
    print(f"  slopes: quantum {sq:.2f}, MC {sc:.2f}; at {q_cost[-1]:.2e} calls MC needs {n_match:.2e} particles")

    # real Q^k A circuits on Aer (tiny burst, 19 qubits)
    s = tiny()
    q = build_qps_circuit(s)
    region = [(1, 2), (2, 2), (1, 3)]
    A, obj = q.state_preparation(region, None)
    sampler = AerAmplitudeSampler(A, obj, seed=7)
    a = exact_amplitude(s, region, None)
    aer_eps = [3e-2, 1e-2, 3e-3] if quick else [3e-2, 1e-2, 3e-3, 1e-3]
    aer_trials = 4 if quick else 8
    aer_cost, aer_rmse, aer_powers = [], [], 0
    t0 = time.time()
    for eps in aer_eps:
        runs = [iqae(sampler, eps, alpha=ALPHA, shots=SHOTS) for _ in range(aer_trials)]
        aer_cost.append(float(np.mean([r.queries for r in runs])))
        aer_rmse.append(float(np.sqrt(np.mean([(r.estimate - a) ** 2 for r in runs]))))
        aer_powers = max(aer_powers, max(max(r.powers) for r in runs))
    print(f"  Aer IQAE on {A.num_qubits}-qubit particle circuits: max Grover power {aer_powers}, "
          f"{time.time() - t0:.0f}s")
    # IQAE error is amplitude-dependent; rescale tiny-system errors to the mean per-sqrt(a(1-a)) level
    ref_scale = np.mean([math.sqrt(r["amplitude"] * (1 - r["amplitude"])) for r in rows]) / math.sqrt(a * (1 - a))

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.loglog(q_cost, c_rmse, "o-", color=CLASSICAL, label=f"classical particle sampling (slope {sc:.2f})")
    ax.loglog(q_cost, q_rmse, "o-", color=QUANTUM, label=f"quantum amplitude estimation (slope {sq:.2f})")
    ax.loglog(aer_cost, np.array(aer_rmse) * ref_scale, "D", mfc="none", color=QUANTUM, ms=8,
              label=f"same, real $Q^kA$ circuits on Aer ({A.num_qubits} qubits)")
    x = np.array([q_cost[0], q_cost[-1]])
    ax.loglog(x, c_rmse[0] * (x / x[0]) ** -0.5, ":", color=MUTED, lw=1)
    ax.loglog(x, q_rmse[0] * (x / x[0]) ** -1.0, ":", color=MUTED, lw=1)
    ax.set_xlabel("particle-program evaluations  (calls of A  /  particles simulated)")
    ax.set_ylabel("RMSE of region brightness")
    ax.set_title("Measuring a particle system: error vs. cost (4 region queries)", loc="left")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig7_qps_convergence.png", dpi=160)
    plt.close(fig)
    return dict(
        queries=rows, epsilons=epsilons, trials=trials, quantum_calls=q_cost.tolist(),
        quantum_rmse=q_rmse.tolist(), mc_rmse=c_rmse.tolist(), slope_quantum=sq, slope_mc=sc,
        mc_particles_to_match_best=float(n_match), speedup_at_best=float(n_match / q_cost[-1]),
        aer=dict(qubits=A.num_qubits, epsilons=aer_eps, calls=aer_cost, rmse=aer_rmse, max_power=aer_powers,
                 amplitude=a),
    )


def _sensor_grid(s, cells: int):
    k = s.pixels // cells
    return [tile(i * k, j * k, k, k) for j in range(cells) for i in range(cells)]


def exp_sensor_grid(out: Path, quick: bool) -> dict:
    """Fig. 8: an 8x8 grid of light sensors over the firework, equal total budget.

    Every sensor needs its own amplitude estimation, while one classical particle
    lands in (at most) one sensor and updates all of them.  The fair comparison
    therefore gives Monte Carlo the *total* quantum budget.
    """
    s = firework(8)
    cells = 8
    regions = _sensor_grid(s, cells)
    img = exact_image(s)
    truth = np.array([[sum(img[c, y, x] for x, y in r) for r in regions] for c in range(3)])
    rng = np.random.default_rng(2)
    epsilons = [1e-3, 1e-4, 1e-5] if quick else [1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
    rows, est_q, est_c = [], {}, {}
    # law of a single particle's (sensor, rgb) contribution: categorical over sensors x ages
    from .particles import exact_age_density

    dens = exact_age_density(s)[0]  # burst: one age
    p_sensor = np.array([sum(dens[y, x] for x, y in r) for r in regions])
    inten = s.intensity(s.ages())[:, 0]
    for eps in epsilons:
        q = np.zeros_like(truth)
        calls = 0
        for c in range(3):
            for i, a in enumerate(truth[c]):
                r = iqae(ExactAmplitudeSampler(a, rng), eps, alpha=ALPHA, shots=SHOTS)
                q[c, i] = r.estimate
                calls += r.queries
        counts = rng.multinomial(calls, np.append(p_sensor, max(0.0, 1 - p_sensor.sum())))[:-1]
        mc = inten[:, None] * counts[None, :] / calls
        rq, rc = float(np.sqrt(np.mean((q - truth) ** 2))), float(np.sqrt(np.mean((mc - truth) ** 2)))
        rows.append(dict(epsilon=eps, total_calls=calls, rmse_quantum=rq, rmse_mc=rc, mc_over_quantum=rc / rq))
        est_q[eps], est_c[eps] = q, mc
        print(f"  sensor grid eps={eps:.0e}: {calls:.2e} calls, RMSE quantum {rq:.2e} vs MC {rc:.2e}")

    def as_img(v):
        return v.reshape(3, cells, cells)

    lo, hi = epsilons[0], epsilons[-1]
    with plt.rc_context({"axes.grid": False}):
        fig, axs = plt.subplots(2, 4, figsize=(13, 6.6))
        white = float(as_img(truth).max())
        show(axs[0, 0], img, f"firework t=8, exact\n({s.num_particles:.2g} particles)")
        show(axs[1, 0], as_img(truth), "8x8 sensor grid, exact", white, smooth=False)
        errs = []
        for col, eps in ((1, lo), (2, hi)):
            row = next(r for r in rows if r["epsilon"] == eps)
            show(axs[0, col], as_img(est_q[eps]), f"quantum AE, {row['total_calls']:.1e} calls\n"
                 f"RMSE {row['rmse_quantum']:.1e}", white, smooth=False)
            show(axs[1, col], as_img(est_c[eps]), f"Monte Carlo, same {row['total_calls']:.1e} particles\n"
                 f"RMSE {row['rmse_mc']:.1e}", white, smooth=False)
            errs.append((np.abs(est_q[eps] - truth).max(0), np.abs(est_c[eps] - truth).max(0)))
        ax = axs[0, 3]
        ax.set_axis_on()
        calls = [r["total_calls"] for r in rows]
        ax.loglog(calls, [r["rmse_mc"] for r in rows], "o-", color=CLASSICAL, label="Monte Carlo")
        ax.loglog(calls, [r["rmse_quantum"] for r in rows], "o-", color=QUANTUM, label="quantum AE")
        ax.set_xlabel("total budget (calls / particles)")
        ax.set_ylabel("RMSE per sensor")
        ax.set_title("64 sensors, equal total budget", fontsize=9, fontweight="normal")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True)
        ax = axs[1, 3]
        e_q, e_c = errs[-1]
        vmax = max(e_q.max(), e_c.max())
        ax.imshow(np.hstack([e_q.reshape(cells, cells), np.full((cells, 1), np.nan), e_c.reshape(cells, cells)]),
                  origin="lower", cmap="Blues", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(f"|error| at {hi:.0e}: quantum (left) vs MC", fontsize=9, fontweight="normal")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.suptitle("Rendering many regions at once: quantum wins only past a crossover budget",
                     x=0.01, ha="left", fontweight="bold")
        fig.tight_layout()
        fig.savefig(out / "fig8_qps_sensor_grid.png", dpi=150)
        plt.close(fig)
    return dict(sensors=len(regions), rows=rows)


def exp_bounds_events(out: Path, quick: bool) -> dict:
    """Fig. 9: bounding box (Duerr-Hoyer) and event search (BBHT) vs. particle count."""
    rng = np.random.default_rng(3)
    steps = [2, 3, 4, 5, 6, 7] if quick else [2, 3, 4, 5, 6, 7, 8]
    trials = 10 if quick else 25
    bb_rows = []
    for t in steps:
        s = firework(t)
        N = s.num_particles
        r = simulate(s, np.arange(N))
        sides = {"min x": r["x"], "max x": -r["x"], "min y": r["y"], "max y": -r["y"]}
        costs, ok = [], 0
        for _ in range(trials):
            total = 0
            for v in sides.values():
                c = QueryCounter()
                i = minimum_finding(v, np.ones(N, bool), ExactGroverBackend(rng), rng, c)
                ok += int(v[i] == v.min())
                total += c.total
            costs.append(total)
        bb_rows.append(dict(N=N, quantum_evals=float(np.mean(costs)), classical_evals=N,
                            exact_fraction=ok / (4 * trials)))
        print(f"  bounding box N={N:>9,}: quantum {np.mean(costs):8.0f} evals, exact {ok / (4 * trials):.2f}")

    # event search: "find a spark that hits this target" for targets of decreasing size
    s = firework(8)
    r = simulate(s, np.arange(s.num_particles))
    pix = np.where(r["visible"], r["py"] * s.pixels + r["px"], -1)
    counts = np.bincount(pix[pix >= 0], minlength=s.pixels**2)
    rng_t = np.random.default_rng(4)
    fracs = [2.0**-k for k in range(4, 23, 2)]
    ev_rows = []
    for f in fracs:
        # target = the pixel whose share of all particles is closest to f
        target_pixel = int(np.argmin(np.abs(np.log(np.maximum(counts, 0.5) / s.num_particles) - np.log(f))))
        mask = pix == target_pixel
        M = int(mask.sum())
        if M == 0:
            continue
        costs, found = [], 0
        for _ in range(trials):
            c = QueryCounter()
            i = None
            while i is None:  # restart a failed QSearch (rare)
                i = qsearch(mask, ExactGroverBackend(rng_t), rng_t, c)
            found += int(mask[i])
            costs.append(c.total)
        ev_rows.append(dict(N=s.num_particles, M=M, ratio=s.num_particles / M, quantum_evals=float(np.mean(costs)),
                            classical_evals=float((s.num_particles + 1) / (M + 1)), success=found / trials))
    ev_rows = sorted({row["M"]: row for row in ev_rows}.values(), key=lambda d: d["ratio"])
    for row in ev_rows:
        print(f"  event search N/M={row['ratio']:>10.0f}: quantum {row['quantum_evals']:7.0f}, "
              f"classical {row['classical_evals']:9.0f}")

    N = np.array([r_["N"] for r_ in bb_rows], float)
    qb = np.array([r_["quantum_evals"] for r_ in bb_rows])
    ratio = np.array([r_["ratio"] for r_ in ev_rows])
    qe = np.array([r_["quantum_evals"] for r_ in ev_rows])
    ce = np.array([r_["classical_evals"] for r_ in ev_rows])
    s_bb, s_ev = _loglog_slope(N, qb), _loglog_slope(ratio, qe)
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axs[0]
    ax.loglog(N, 4 * N, "o-", color=CLASSICAL, label="classical: simulate every particle (4 sides)")
    ax.loglog(N, qb, "o-", color=QUANTUM, label=f"quantum Dürr–Høyer (slope {s_bb:.2f})")
    ax.set_xlabel("particles N")
    ax.set_ylabel("particle-program evaluations")
    ax.set_title("Exact bounding box of N particles", loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax = axs[1]
    ax.loglog(ratio, ce, "o-", color=CLASSICAL, label="classical: sample particles until a hit")
    ax.loglog(ratio, qe, "o-", color=QUANTUM, label=f"quantum BBHT search (slope {s_ev:.2f})")
    ax.set_xlabel("rarity N / M  (particles per hitting particle)")
    ax.set_ylabel("particle-program evaluations")
    ax.set_title("Find a particle that triggers an event", loc="left")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig9_qps_bounds_events.png", dpi=160)
    plt.close(fig)
    return dict(bounding_box=bb_rows, bounding_box_slope=s_bb, events=ev_rows, events_slope=s_ev)


def circuit_stats() -> dict:
    """Gate and qubit counts of U_particles (independent of the particle count)."""
    stats = {}
    for s in (tiny(), firework(8), firework(12), fountain()):
        q = build_qps_circuit(s)
        ops = q.dynamics.count_ops()
        mcx = sum(v for k, v in ops.items() if k.startswith(("mcx", "ccx", "cx")))
        stats[f"{s.name}{'' if s.stream else f'(t={s.steps})'}"] = dict(
            qubits=q.num_qubits, seed_qubits=s.seed_bits, particles=float(s.num_particles),
            gates=q.dynamics.size(), controlled_x=mcx, steps=s.steps,
            max_controls=max((op.operation.num_qubits - 1 for op in q.dynamics.data), default=0),
        )
    return stats


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--only", nargs="*", default=None, help="subset: frames convergence sensors bounds stats")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "qps_results.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    todo = {
        "stats": lambda: circuit_stats(),
        "frames": lambda: exp_frames(out, args.quick),
        "convergence": lambda: exp_convergence(out, args.quick),
        "sensors": lambda: exp_sensor_grid(out, args.quick),
        "bounds": lambda: exp_bounds_events(out, args.quick),
    }
    for name, fn in todo.items():
        if args.only and name not in args.only:
            continue
        print(f"[{name}]")
        t0 = time.time()
        results[name] = fn()
        print(f"  done in {time.time() - t0:.1f}s")
        path.write_text(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()

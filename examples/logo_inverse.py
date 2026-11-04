"""Inverse problem with many unknowns: the initial velocity of every cell that turns uniform
gas into a logo at t = 1.

The gas starts at density 1 and pressure 0.01 in a periodic unit box, on a moving
mesh. The unknowns are the two velocity components of each cell, 2N numbers. The target is
a density that is three times higher along thick lines tracing the periodic Voronoi diagram
of six sites than between them, with the same mean. The loss is the area-weighted squared
mismatch at t = 1. L-BFGS starts from gas at rest, with reverse-mode gradients through the
whole run.

Usage: uv run --group dev python examples/logo_inverse.py [--k 48] [--iters 150]
Writes runs/logo_inverse/: velocity.npy, history.npy, logo_inverse.png, movie.mp4, movie.gif.
"""

import argparse
import functools
import math
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from scipy.optimize import minimize

import dvf  # noqa: F401  (enables float64)
from dvf import hydro, mesh, plot, problems, solver

BOX, T_END = 1.0, 1.0
# Cold gas: the flow coasts to the logo, with pressure only resisting at the end.
P0 = 0.01
# Flow speeds the time step allows for, and the weight of the smoothness penalty.
V_MAX, SMOOTH = 0.5, 1e-3
SITES = np.random.default_rng(1).random((6, 2))
OUT = Path("runs/logo_inverse")


def target(c):
    return problems.voronoi_logo(c % BOX, SITES, BOX)


def setup(k):
    cfg = solver.Config(box=BOX)
    pos = problems.lattice(k, BOX, jitter=0.3, seed=2)
    area = mesh.geometry(pos, mesh.topology(pos, BOX), BOX).area

    def start(v):
        n = pos.shape[0]
        w = jnp.concatenate([jnp.ones((n, 1)), v.reshape(n, 2), jnp.full((n, 1), P0)], axis=1)
        return solver.State(pos, hydro.prim_to_cons(w, cfg.gamma) * area[:, None])

    # The gas is cold, so the flow sets the time step: cells meet at up to twice V_MAX.
    c0 = math.sqrt(cfg.gamma * P0)
    radius = float(jnp.sqrt(area.min() / jnp.pi))
    n_steps = math.ceil(T_END / (0.3 * radius / (c0 + 2.0 * V_MAX)))
    return cfg, pos, start, T_END / n_steps, n_steps


def solve(k, iters):
    cfg, pos, start, dt, n_steps = setup(k)
    segment = next(s for s in range(int(math.sqrt(n_steps)), 0, -1) if n_steps % s == 0)

    topo0 = mesh.topology(pos, BOX)
    geo0 = mesh.geometry(pos, topo0, BOX)

    def loss(v):
        s = solver.run(start(v), dt, n_steps, cfg, segment)
        geo = mesh.geometry(s.pos, mesh.topology(s.pos, BOX), BOX)
        rho = s.q[:, 0] / geo.area
        misfit = jnp.sum(geo.area * (rho - target(geo.centroid)) ** 2)
        # Penalize differences between neighbors' starting velocities, so that the field
        # stays smooth on the scale of the cells.
        vv = v.reshape(-1, 2)
        jump = jnp.sum((vv[topo0.a] - vv[topo0.b]) ** 2, axis=1)
        return misfit + SMOOTH * jnp.sum(geo0.length * jump / geo0.dist)

    value_and_grad = jax.jit(jax.value_and_grad(loss))
    history = []
    t0 = time.time()

    def fun(x):
        v, g = value_and_grad(jnp.asarray(x))
        if not np.isfinite(float(v)):
            return 1e10, np.zeros_like(x)
        history.append(float(v))
        if len(history) % 5 == 1:
            speed = np.abs(x).max()
            print(
                f"  eval {len(history):4d}  loss {float(v):.4e}  max |v| {speed:.3f}  ({time.time() - t0:.0f} s)",
                flush=True,
            )
        return float(v), np.asarray(g)

    n = pos.shape[0]
    print(f"{n} cells, {2 * n} unknowns, {n_steps} steps of {dt:.2e}")
    result = minimize(
        fun, np.zeros(2 * n), jac=True, method="L-BFGS-B", options={"maxiter": iters, "maxcor": 20}
    )
    print(
        f"loss {history[0]:.4e} -> {history[-1]:.4e} in {len(history)} evaluations, {time.time() - t0:.0f} s"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "velocity.npy", result.x)
    np.save(OUT / "history.npy", np.array(history))


def snapshots(k, frames):
    """States at `frames` evenly spaced times, with the optimized velocities."""
    cfg, _pos, start, dt, n_steps = setup(k)
    v = jnp.asarray(np.load(OUT / "velocity.npy"))
    per = n_steps // (frames - 1)
    advance = jax.jit(functools.partial(solver.run, dt=dt, n_steps=per, cfg=cfg))
    s = start(v)
    out = [(0.0, s)]
    for f in range(1, frames):
        s = advance(s)
        out.append((f * per * dt, s))
    if per * (frames - 1) < n_steps:
        s = jax.jit(
            functools.partial(solver.run, dt=dt, n_steps=n_steps - per * (frames - 1), cfg=cfg)
        )(s)
        out[-1] = (T_END, s)
    return [(t, np.asarray(s.pos), np.asarray(density(s))) for t, s in out]


def density(s):
    geo = mesh.geometry(s.pos, mesh.topology(s.pos, BOX), BOX)
    return s.q[:, 0] / geo.area


def draw_cells(ax, pos, rho, edges=True):
    polys, values = plot.wrapped(plot.cell_polygons(pos, BOX), rho, BOX)
    pc = PolyCollection(
        polys,
        array=values,
        cmap="magma",
        edgecolors=(1, 1, 1, 0.15) if edges else "face",
        linewidths=0.2,
    )
    pc.set_clim(0.4, 2.0)
    ax.add_collection(pc)
    ax.set_xlim(0, BOX)
    ax.set_ylim(0, BOX)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return pc


def figure(k):
    snaps = snapshots(k, 2)
    (_, pos0, _), (_, pos1, rho1) = snaps
    v = np.load(OUT / "velocity.npy").reshape(-1, 2)
    history = np.load(OUT / "history.npy")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), dpi=100)
    g = (np.arange(256) + 0.5) / 256
    gx, gy = np.meshgrid(g, g)
    t_img = np.asarray(target(jnp.asarray(np.c_[gx.ravel(), gy.ravel()]))).reshape(256, 256)
    axes[0].imshow(t_img, origin="lower", extent=[0, 1, 0, 1], cmap="magma", vmin=0.4, vmax=2.0)
    axes[0].set_title("target density at t = 1")
    speed = np.linalg.norm(v, axis=1)
    polys, values = plot.wrapped(plot.cell_polygons(pos0, BOX), speed, BOX)
    pc = PolyCollection(polys, array=values, cmap="viridis", edgecolors="face")
    axes[1].add_collection(pc)
    step = max(1, round(math.sqrt(len(v)) / 24))
    kk = round(math.sqrt(len(v)))
    sel = np.arange(len(v)).reshape(kk, kk)[::step, ::step].ravel()
    axes[1].quiver(
        pos0[sel, 0], pos0[sel, 1], v[sel, 0], v[sel, 1], color="w", scale=4.0, width=0.004
    )
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_aspect("equal")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.colorbar(pc, ax=axes[1], shrink=0.8, label="initial speed")
    axes[1].set_title("recovered initial velocity (uniform density)")
    pc = draw_cells(axes[2], pos1, rho1)
    fig.colorbar(pc, ax=axes[2], shrink=0.8, label="density")
    axes[2].set_title("density at t = 1, with the mesh")
    axes[3].semilogy(history, "k-")
    axes[3].set_xlabel("evaluation")
    axes[3].set_ylabel("loss")
    axes[3].set_title(f"L-BFGS over {len(v) * 2} unknowns")
    fig.tight_layout()
    fig.savefig(OUT / "logo_inverse.png", dpi=110)
    print(OUT / "logo_inverse.png")


def render(job):
    path, t, pos, rho = job
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    draw_cells(ax, pos, rho)
    ax.text(0.02, 0.97, f"t = {t:.2f}", color="w", fontsize=14, va="top", transform=ax.transAxes)
    fig.savefig(path)
    plt.close(fig)


def movie(k, frames=61):
    snaps = snapshots(k, frames)
    out = OUT / "frames"
    out.mkdir(exist_ok=True)
    for stale in out.glob("*.png"):
        stale.unlink()
    with Pool() as pool:
        pool.map(
            render, [(out / f"frame_{i:04d}.png", t, p, r) for i, (t, p, r) in enumerate(snaps)]
        )
    pattern = str(out / "frame_%04d.png")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            "15",
            "-i",
            pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(OUT / "movie.mp4"),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            "15",
            "-i",
            pattern,
            "-vf",
            "scale=360:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse",
            str(OUT / "movie.gif"),
        ],
        check=True,
    )
    print(OUT / "movie.gif")


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=48, help="cells per side")
    parser.add_argument("--iters", type=int, default=150, help="L-BFGS iterations")
    parser.add_argument("--plot-only", action="store_true", help="reuse the saved velocities")
    parser.add_argument("--out", type=Path, default=OUT, help="output directory")
    args = parser.parse_args()
    OUT = args.out
    if not args.plot_only:
        solve(args.k, args.iters)
    figure(args.k)
    movie(args.k)


if __name__ == "__main__":
    main()

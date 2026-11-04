"""Inverse problem: recover an isentropic vortex's strength and starting position from the
density field it leaves at a later time.

The vortex rides a uniform stream across a periodic box of side 10 on a moving mesh. The
observation is the exact density at t = 2 for strength 5 starting at (5, 5); the loss is
its area-weighted mismatch with the simulated density over the cells. L-BFGS minimizes it
from a wrong first guess, with gradients from reverse mode through the whole run, the mesh
motion included.

Usage: uv run python examples/vortex_inverse.py [cells per side, default 32]
Writes runs/vortex_inverse/inverse.png.
"""

import math
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import dvf  # noqa: F401  (enables float64)
from dvf import mesh, problems, solver

BOX, T_END = 10.0, 2.0
TRUTH = np.array([5.0, 5.0, 5.0])  # strength, x0, y0
GUESS = np.array([3.0, 4.2, 5.7])
LEVELS = np.linspace(0.45, 1.0, 23)


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    cfg = solver.Config(box=BOX)
    pos = problems.lattice(k, BOX)

    def start(theta):
        init = lambda c: problems.isentropic_vortex(c, 0.0, BOX, theta[0], (theta[1], theta[2]))
        return solver.initial_state(pos, init, cfg)

    n_steps = math.ceil(T_END / solver.stable_dt(start(jnp.asarray(TRUTH)), cfg))
    dt = T_END / n_steps

    def final_density(theta):
        s = solver.run(start(theta), dt, n_steps, cfg)
        geo = mesh.geometry(s.pos, mesh.topology(s.pos, BOX), BOX)
        return geo, s.q[:, 0] / geo.area

    def loss(theta):
        geo, rho = final_density(theta)
        observed = problems.isentropic_vortex(
            geo.centroid % BOX, T_END, BOX, TRUTH[0], tuple(TRUTH[1:])
        )[:, 0]
        return jnp.sum(geo.area * (rho - observed) ** 2) / BOX**2

    value_and_grad = jax.jit(jax.value_and_grad(loss))
    history = []

    def fun(theta):
        v, g = value_and_grad(jnp.asarray(theta))
        history.append((float(v), np.array(theta)))
        print(
            f"  eval {len(history):3d}  loss {float(v):.3e}  theta {np.round(theta, 4)}  |grad| {float(jnp.linalg.norm(g)):.2e}"
        )
        return float(v), np.asarray(g)

    print(f"{k}^2 cells, {n_steps} steps; truth {TRUTH}, first guess {GUESS}")
    t0 = time.time()
    result = minimize(
        fun,
        GUESS,
        jac=True,
        method="L-BFGS-B",
        options={"gtol": 1e-10, "ftol": 1e-14, "maxiter": 60},
    )
    print(
        f"recovered {np.round(result.x, 4)} in {len(history)} evaluations, {time.time() - t0:.1f} s ({result.message})"
    )

    out = Path("runs/vortex_inverse")
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4), dpi=100)
    for ax, theta, title in [
        (axes[0], GUESS, "first guess"),
        (axes[1], result.x, "recovered"),
    ]:
        geo, rho = final_density(jnp.asarray(theta))
        c = np.asarray(geo.centroid) % BOX
        sc = ax.tricontourf(
            c[:, 0], c[:, 1], np.asarray(rho), levels=LEVELS, cmap="viridis", extend="both"
        )
        ax.set_title(
            f"{title}\nstrength {theta[0]:.3f}, start ({theta[1]:.3f}, {theta[2]:.3f})", fontsize=9
        )
        ax.set_aspect("equal")
    c = np.asarray(geo.centroid) % BOX
    exact = problems.isentropic_vortex(jnp.asarray(c), T_END, BOX, TRUTH[0], tuple(TRUTH[1:]))[:, 0]
    axes[2].tricontourf(
        c[:, 0], c[:, 1], np.asarray(exact), levels=LEVELS, cmap="viridis", extend="both"
    )
    axes[2].set_title("observed (exact)\nstrength 5, start (5, 5)", fontsize=9)
    axes[2].set_aspect("equal")
    fig.colorbar(sc, ax=axes[:3], shrink=0.8, label="density at t = 2")
    axes[3].semilogy([h[0] for h in history], "k.-")
    axes[3].set_xlabel("evaluation")
    axes[3].set_ylabel("loss")
    axes[3].set_title("L-BFGS with reverse-mode gradients", fontsize=9)
    fig.savefig(out / "inverse.png", bbox_inches="tight")
    print(out / "inverse.png")


if __name__ == "__main__":
    main()

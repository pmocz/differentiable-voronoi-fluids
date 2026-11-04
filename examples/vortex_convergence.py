"""Accuracy of the forward solver: isentropic vortex carried once diagonally across a periodic
box of side 10 to t = 2, L1 density error against the exact solution.

Usage: uv run python examples/vortex_convergence.py [32 64 128]
"""

import functools
import math
import sys
import time

import jax
import jax.numpy as jnp

import dvf  # noqa: F401  (enables float64)
from dvf import mesh, problems, solver

BOX, T_END = 10.0, 2.0


def main():
    ks = [int(a) for a in sys.argv[1:]] or [32, 64, 128]
    cfg = solver.Config(box=BOX)
    prev = None
    for k in ks:
        pos = problems.lattice(k, BOX)
        state = solver.initial_state(pos, lambda c: problems.isentropic_vortex(c, 0.0, BOX), cfg)
        n_steps = math.ceil(T_END / solver.stable_dt(state, cfg))
        dt = T_END / n_steps
        start = time.time()
        final = jax.jit(functools.partial(solver.run, dt=dt, n_steps=n_steps, cfg=cfg))(state)
        geo = mesh.geometry(final.pos, mesh.topology(final.pos, BOX), BOX)
        rho = final.q[:, 0] / geo.area
        exact = problems.isentropic_vortex(geo.centroid % BOX, T_END, BOX)[:, 0]
        err = float(jnp.sum(jnp.abs(rho - exact) * geo.area) / BOX**2)
        order = f"{math.log(prev / err) / math.log(2):.2f}" if prev else "-"
        print(
            f"{k:4d}^2  steps {n_steps:4d}  L1(rho) {err:.4e}  order {order}  ({time.time() - start:.1f} s)"
        )
        prev = err


if __name__ == "__main__":
    main()

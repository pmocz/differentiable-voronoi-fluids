"""Initial conditions and seed layouts."""

import jax.numpy as jnp
import numpy as np


def lattice(k: int, box: float, jitter: float = 0.0, seed: int = 0) -> jnp.ndarray:
    """k x k seeds at the centers of a square grid, each moved by up to `jitter` cell widths."""
    h = box / k
    xs = (np.arange(k) + 0.5) * h
    x, y = np.meshgrid(xs, xs)
    pos = np.c_[x.ravel(), y.ravel()]
    if jitter:
        pos += jitter * h * (np.random.default_rng(seed).random(pos.shape) - 0.5)
    return jnp.asarray(pos % box)


def isentropic_vortex(c, t, box, strength=5.0, center=(5.0, 5.0), drift=(1.0, 1.0), gamma=1.4):
    """Yee et al. isentropic vortex, carried by a uniform `drift` across a periodic box.
    Exact at any time: [rho, vx, vy, p] at points `c`."""
    center = jnp.asarray(center) + jnp.asarray(drift) * t
    d = c - center
    d = d - box * jnp.round(d / box)
    r2 = jnp.sum(d * d, axis=-1)
    f = strength / (2.0 * jnp.pi) * jnp.exp(0.5 * (1.0 - r2))
    temp = 1.0 - (gamma - 1.0) * strength**2 / (8.0 * gamma * jnp.pi**2) * jnp.exp(1.0 - r2)
    rho = temp ** (1.0 / (gamma - 1.0))
    return jnp.stack([rho, drift[0] - f * d[:, 1], drift[1] + f * d[:, 0], rho**gamma], axis=-1)

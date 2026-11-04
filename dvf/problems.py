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


def _distance_to_voronoi_edges(c, sites, box):
    """Distance from each point to the edge of its cell in the periodic Voronoi diagram of
    `sites`: the nearest bisector between its own site and any other."""
    shifts = jnp.array([[i, j] for i in (-1, 0, 1) for j in (-1, 0, 1)], dtype=c.dtype) * box
    images = (sites[None, :, :] + shifts[:, None, :]).reshape(-1, 2)
    d2 = jnp.sum((c[:, None, :] - images[None, :, :]) ** 2, axis=-1)
    own = images[jnp.argmin(d2, axis=1)]
    sep = jnp.linalg.norm(images[None, :, :] - own[:, None, :], axis=-1)
    to_bisector = (d2 - jnp.min(d2, axis=1, keepdims=True)) / (
        2.0 * jnp.where(sep > 1e-12, sep, 1.0)
    )
    return jnp.min(jnp.where(sep > 1e-12, to_bisector, jnp.inf), axis=1)


def voronoi_logo(c, sites, box=1.0, width=0.04, contrast=3.0):
    """Density that is high along thick lines tracing the periodic Voronoi diagram of
    `sites`, with mean 1 over the box. The lines have profile exp(-(d / width)^2) across
    them and are `contrast` times as dense as the background."""
    sites = jnp.asarray(sites, dtype=c.dtype)

    def bump(points):
        return jnp.exp(-((_distance_to_voronoi_edges(points, sites, box) / width) ** 2))

    g = (jnp.arange(512) + 0.5) / 512 * box
    gx, gy = jnp.meshgrid(g, g)
    mean_bump = jnp.mean(bump(jnp.stack([gx.ravel(), gy.ravel()], axis=1)))
    return (1.0 + (contrast - 1.0) * bump(c)) / (1.0 + (contrast - 1.0) * mean_bump)

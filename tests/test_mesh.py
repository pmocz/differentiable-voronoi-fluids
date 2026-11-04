import jax
import jax.numpy as jnp
import numpy as np
import pytest

import dvf  # noqa: F401
from dvf import mesh, problems


@pytest.mark.parametrize("n", [3, 50, 2000])
def test_random_cells_tile_the_box(n):
    box = 2.0
    pos = jnp.asarray(np.random.default_rng(n).random((n, 2)) * box)
    topo = mesh.topology(pos, box)
    geo = mesh.geometry(pos, topo, box)
    assert topo.a.shape == (3 * n,)
    assert abs(float(geo.area.sum()) - box * box) < 1e-12
    assert bool(jnp.all(geo.area > 0))


def test_lattice_cells_are_squares():
    # Cocircular seeds: the triangulation picks diagonals, whose faces have zero length.
    box, k = 3.0, 12
    pos = problems.lattice(k, box)
    geo = mesh.geometry(pos, mesh.topology(pos, box), box)
    h = box / k
    assert float(jnp.abs(geo.area - h * h).max()) < 1e-12
    assert float(jnp.abs(geo.centroid - pos).max()) < 1e-12


def test_geometry_gradient_matches_finite_differences():
    # With the connectivity held, area is a smooth function of the seeds.
    box = 1.0
    pos = jnp.asarray(np.random.default_rng(3).random((40, 2)))
    topo = mesh.topology(pos, box)

    def f(p):
        geo = mesh.geometry(p, topo, box)
        return jnp.sum(geo.area**2 * jnp.sin(geo.centroid[:, 0]))

    g = jax.grad(f)(pos)
    v = jnp.asarray(np.random.default_rng(4).normal(size=pos.shape))
    eps = 1e-6
    fd = (f(pos + eps * v) - f(pos - eps * v)) / (2 * eps)
    # The directional derivative is a small sum of large terms; compare to their size.
    assert abs(float(fd - jnp.sum(g * v))) < 1e-7 * float(jnp.sum(jnp.abs(g * v)))

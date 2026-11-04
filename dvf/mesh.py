"""Periodic Voronoi mesh: connectivity on the host, geometry in JAX.

Which seeds are neighbors is a piecewise-constant function of their positions, so it is
computed outside JAX, from gradient-stopped positions, by SciPy's Delaunay triangulation.
Everything with a size (Voronoi vertices, face lengths and normals, cell areas and
centroids) is then computed in JAX from the positions and that connectivity, so gradients
flow through it. A Delaunay triangulation of N points on a torus has exactly 3N edges, so
the face arrays always have the same shape and nothing recompiles when the mesh flips.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import Delaunay


class Topology(NamedTuple):
    """Integer connectivity of the 3N Voronoi faces.

    Face f separates cell `a[f]` from the image of cell `b[f]` displaced by `b_off[f]` box
    lengths. Its two ends are the circumcenters of the Delaunay triangles on either side of
    the edge, whose corners are cells `tri[f, k]` displaced by `tri_off[f, k]`, k = 0, 1.
    All offsets are relative to cell a's position in the box.
    """

    a: jax.Array  # (F,)
    b: jax.Array  # (F,)
    b_off: jax.Array  # (F, 2)
    tri: jax.Array  # (F, 2, 3)
    tri_off: jax.Array  # (F, 2, 3, 2)


class Geometry(NamedTuple):
    length: jax.Array  # (F,) face length
    mid: jax.Array  # (F, 2) face midpoint, in cell a's frame
    normal: jax.Array  # (F, 2) unit normal from a to b
    dist: jax.Array  # (F,) distance between the two seeds
    area: jax.Array  # (N,)
    centroid: jax.Array  # (N, 2), in each cell's own frame


def _topology_host(pos: np.ndarray, box: float) -> tuple[np.ndarray, ...]:
    pos = np.asarray(pos, dtype=np.float64)
    n = len(pos)
    spacing = box / np.sqrt(n)
    # Cocircular seeds, as on a lattice, leave the triangulation a free choice that Qhull
    # may make differently in different periodic copies. A tiny nudge, the same for every
    # copy of a seed, breaks the tie consistently; the faces it decides have zero length.
    nudge = 1e-7 * spacing * (np.random.default_rng(0).random((n, 2)) - 0.5)
    pos = pos + nudge
    buffer = 4.0 * spacing
    while True:
        out = _try_topology(pos, box, buffer)
        if out is not None:
            return out
        buffer *= 1.5
        if buffer > box:
            raise RuntimeError("periodic Delaunay triangulation failed")


def _try_topology(pos: np.ndarray, box: float, buffer: float):
    n = len(pos)
    idx = [np.arange(n)]
    off = [np.zeros((n, 2), dtype=np.int64)]
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            if ox == 0 and oy == 0:
                continue
            p = pos + box * np.array([ox, oy])
            keep = np.all((p > -buffer) & (p < box + buffer), axis=1)
            idx.append(np.nonzero(keep)[0])
            off.append(np.tile([ox, oy], (keep.sum(), 1)))
    idx = np.concatenate(idx)
    off = np.concatenate(off)
    pts = pos[idx] + box * off
    simplices = Delaunay(pts).simplices

    # Each edge with the two triangles that share it.
    m = len(simplices)
    edges = np.concatenate([simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [2, 0]]])
    owner = np.tile(np.arange(m), 3)
    edges.sort(axis=1)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, owner = edges[order], owner[order]
    same = np.all(edges[1:] == edges[:-1], axis=1)
    first = np.nonzero(same)[0]
    u, v = edges[first, 0], edges[first, 1]
    t1, t2 = owner[first], owner[first + 1]

    # Keep each face once, seen from a cell in the box: an edge between two cells in the box,
    # or between a cell in the box and an image of a higher-numbered cell (or of itself,
    # displaced in the positive direction).
    u_in = np.all(off[u] == 0, axis=1)
    v_in = np.all(off[v] == 0, axis=1)
    swap = ~u_in & v_in
    a_t = np.where(swap, v, u)
    b_t = np.where(swap, u, v)
    a_in = u_in | v_in
    ia, ib = idx[a_t], idx[b_t]
    ob = off[b_t]
    positive = (ob[:, 0] > 0) | ((ob[:, 0] == 0) & (ob[:, 1] > 0))
    keep = a_in & ((ia < ib) | ((ia == ib) & positive))
    if keep.sum() != 3 * n:
        return None
    a_t, b_t, t1, t2 = a_t[keep], b_t[keep], t1[keep], t2[keep]

    tri = np.stack([simplices[t1], simplices[t2]], axis=1)  # tiled indices (F, 2, 3)
    return (
        idx[a_t].astype(np.int32),
        idx[b_t].astype(np.int32),
        off[b_t].astype(np.int32),
        idx[tri].astype(np.int32),
        off[tri].astype(np.int32),
    )


def topology(pos: jax.Array, box: float) -> Topology:
    """Connectivity of the periodic Voronoi mesh of `pos`. Carries no gradient."""
    n = pos.shape[0]
    f = 3 * n
    shapes = (
        jax.ShapeDtypeStruct((f,), jnp.int32),
        jax.ShapeDtypeStruct((f,), jnp.int32),
        jax.ShapeDtypeStruct((f, 2), jnp.int32),
        jax.ShapeDtypeStruct((f, 2, 3), jnp.int32),
        jax.ShapeDtypeStruct((f, 2, 3, 2), jnp.int32),
    )
    out = jax.pure_callback(
        lambda p: _topology_host(p, box),
        shapes,
        jax.lax.stop_gradient(pos),
        vmap_method="sequential",
    )
    return Topology(*out)


def safe_norm(v: jax.Array) -> jax.Array:
    """Euclidean norm along the last axis, with a finite gradient at zero."""
    return jnp.sqrt(jnp.sum(v * v, axis=-1) + 1e-300)


def circumcenter(p: jax.Array) -> jax.Array:
    """Circumcenters of triangles with corners p[..., k, :], k = 0, 1, 2."""
    b = p[..., 1, :] - p[..., 0, :]
    c = p[..., 2, :] - p[..., 0, :]
    d = 2.0 * (b[..., 0] * c[..., 1] - b[..., 1] * c[..., 0])
    b2 = jnp.sum(b * b, axis=-1)
    c2 = jnp.sum(c * c, axis=-1)
    ux = (c[..., 1] * b2 - b[..., 1] * c2) / d
    uy = (b[..., 0] * c2 - c[..., 0] * b2) / d
    return p[..., 0, :] + jnp.stack([ux, uy], axis=-1)


def geometry(pos: jax.Array, topo: Topology, box: float) -> Geometry:
    n = pos.shape[0]
    xa = pos[topo.a]
    xb = pos[topo.b] + box * topo.b_off
    corners = pos[topo.tri] + box * topo.tri_off  # (F, 2, 3, 2)
    v = circumcenter(corners)  # (F, 2, 2): the face's two ends
    d = xb - xa
    dist = safe_norm(d)
    length = safe_norm(v[:, 1] - v[:, 0])
    mid = 0.5 * (v[:, 0] + v[:, 1])

    # Each face and seed span a triangle of area length * dist / 4 in both cells.
    tri_area = 0.25 * length * dist
    area = jax.ops.segment_sum(tri_area, topo.a, n) + jax.ops.segment_sum(tri_area, topo.b, n)
    ends = v[:, 0] + v[:, 1]
    ca = (ends - 2.0 * xa) * (tri_area / 3.0)[:, None]
    cb = (ends - 2.0 * xb) * (tri_area / 3.0)[:, None]
    csum = jax.ops.segment_sum(ca, topo.a, n) + jax.ops.segment_sum(cb, topo.b, n)
    centroid = pos + csum / area[:, None]
    return Geometry(length, mid, d / dist[:, None], dist, area, centroid)

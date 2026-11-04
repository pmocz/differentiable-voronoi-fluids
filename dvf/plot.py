"""Cell polygons for plotting, from SciPy's Voronoi diagram of the periodically tiled seeds."""

import numpy as np
from scipy.spatial import Voronoi


def cell_polygons(pos, box: float) -> list[np.ndarray]:
    """The Voronoi cell of each seed, unwrapped around the seed."""
    pos = np.asarray(pos)
    n = len(pos)
    shifts = np.array([[i, j] for i in (0, -1, 1) for j in (0, -1, 1)], dtype=float) * box
    vor = Voronoi(np.concatenate([pos + s for s in shifts]))
    return [vor.vertices[vor.regions[vor.point_region[i]]] for i in range(n)]


def wrapped(polys: list[np.ndarray], values, box: float):
    """Polygons and their values, with a copy one box over of each polygon that crosses an
    edge, so that cells drawn on [0, box)^2 fill it."""
    out, vals = [], []
    for p, v in zip(polys, values):
        lo, hi = p.min(axis=0), p.max(axis=0)
        offsets = [[0.0] + [-box] * int(hi[k] > box) + [box] * int(lo[k] < 0) for k in (0, 1)]
        for dx in offsets[0]:
            for dy in offsets[1]:
                out.append(p + np.array([dx, dy]))
                vals.append(v)
    return out, np.array(vals)

"""Moving-mesh finite-volume update in a periodic box (Springel 2010; the two-stage Heun
scheme of Pakmor et al. 2016), written so that JAX can differentiate a whole run.

The mesh is rebuilt twice per step through `mesh.topology`, whose connectivity carries no
gradient; everything else (geometry, gradients and limiter, face velocities, fluxes, the
seed motion) is differentiated. The time step is fixed per run.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from dvf import hydro, mesh


class Config(NamedTuple):
    box: float
    gamma: float = 1.4
    # Seeds drift towards their cell's centroid once the offset exceeds `eta` cell radii,
    # at up to `reg_speed` times the sound speed.
    eta: float = 0.25
    reg_speed: float = 0.5
    moving: bool = True


class State(NamedTuple):
    pos: jax.Array  # (N, 2) seeds, in the box
    q: jax.Array  # (N, 4) conserved totals per cell


def initial_state(pos, init, cfg: Config) -> State:
    """Cells filled with `init(centroid)`, a function returning [rho, vx, vy, p] arrays."""
    geo = mesh.geometry(pos, mesh.topology(pos, cfg.box), cfg.box)
    w = init(geo.centroid)
    return State(pos, hydro.prim_to_cons(w, cfg.gamma) * geo.area[:, None])


def primitives(q, geo: mesh.Geometry, gamma):
    return hydro.cons_to_prim(q / geo.area[:, None], gamma)


def seed_velocity(pos, w, geo: mesh.Geometry, cfg: Config):
    if not cfg.moving:
        return jnp.zeros_like(pos)
    off = geo.centroid - pos
    d = mesh.safe_norm(off)
    radius = jnp.sqrt(geo.area / jnp.pi)
    ramp = jnp.clip((d / (cfg.eta * radius) - 0.9) / 0.2, 0.0, 1.0)
    steer = off / d[:, None] * (cfg.reg_speed * hydro.sound_speed(w, cfg.gamma) * ramp)[:, None]
    return w[:, 1:3] + steer


def limited_gradients(w, pos, topo: mesh.Topology, geo: mesh.Geometry, box):
    """Least-squares gradients of the primitives, limited so that face values stay within
    the range of each cell and its neighbors. Returns (N, 4, 2)."""
    n = w.shape[0]
    a, b = topo.a, topo.b
    d = pos[b] + box * topo.b_off - pos[a]
    wgt = geo.length / (geo.dist * geo.dist)
    outer = wgt[:, None, None] * d[:, :, None] * d[:, None, :]
    s = jax.ops.segment_sum(outer, a, n) + jax.ops.segment_sum(outer, b, n)
    # Seen from b the offset and the difference both flip sign, so both cells get the same term.
    rhs = wgt[:, None, None] * (w[b] - w[a])[:, :, None] * d[:, None, :]
    r = jax.ops.segment_sum(rhs, a, n) + jax.ops.segment_sum(rhs, b, n)  # (N, 4, 2)
    det = s[:, 0, 0] * s[:, 1, 1] - s[:, 0, 1] * s[:, 1, 0]
    det = hydro.safe(det)
    gx = (s[:, 1, 1][:, None] * r[:, :, 0] - s[:, 0, 1][:, None] * r[:, :, 1]) / det[:, None]
    gy = (s[:, 0, 0][:, None] * r[:, :, 1] - s[:, 1, 0][:, None] * r[:, :, 0]) / det[:, None]
    g = jnp.stack([gx, gy], axis=-1)

    lo = jnp.minimum(
        w, jnp.minimum(jax.ops.segment_min(w[b], a, n), jax.ops.segment_min(w[a], b, n))
    )
    hi = jnp.maximum(
        w, jnp.maximum(jax.ops.segment_max(w[b], a, n), jax.ops.segment_max(w[a], b, n))
    )

    def face_psi(cell, at):
        dw = jnp.sum(g[cell] * (at - geo.centroid[cell])[:, None, :], axis=-1)
        # A change this small needs no limiting; skipping it keeps 1 / dw^2 finite in the
        # derivative of the discarded ratio.
        scale = jnp.abs(w[cell]) + jnp.abs(hi[cell]) + jnp.abs(lo[cell])
        small = jnp.abs(dw) <= 1e-14 * scale + 1e-100
        dw = jnp.where(small, 1.0, dw)
        ratio = jnp.where(dw > 0.0, hi[cell] - w[cell], lo[cell] - w[cell]) / dw
        return jnp.where(small, 1.0, ratio)

    psi = jnp.minimum(
        jax.ops.segment_min(face_psi(a, geo.mid), a, n),
        jax.ops.segment_min(face_psi(b, geo.mid - box * topo.b_off), b, n),
    )
    return g * jnp.clip(psi, 0.0, 1.0)[:, :, None]


def rhs(pos, q, vel, topo: mesh.Topology, cfg: Config):
    """Rate of change of the cell totals on the mesh of `pos`, with seeds moving at `vel`."""
    n = pos.shape[0]
    box, gamma = cfg.box, cfg.gamma
    geo = mesh.geometry(pos, topo, box)
    w = primitives(q, geo, gamma)
    g = limited_gradients(w, pos, topo, geo, box)
    a, b = topo.a, topo.b

    xa = pos[a]
    xb = pos[b] + box * topo.b_off
    d = xb - xa
    corr = jnp.sum((vel[a] - vel[b]) * (geo.mid - 0.5 * (xa + xb)), axis=-1) / (geo.dist * geo.dist)
    wf = 0.5 * (vel[a] + vel[b]) + d * corr[:, None]

    def extrapolate(cell, at):
        we = w[cell] + jnp.sum(g[cell] * (at - geo.centroid[cell])[:, None, :], axis=-1)
        ok = (we[:, 0] > 0.0) & (we[:, 3] > 0.0)
        return jnp.where(ok[:, None], we, w[cell])

    wl = extrapolate(a, geo.mid)
    wr = extrapolate(b, geo.mid - box * topo.b_off)
    f = hydro.face_flux(wl, wr, geo.normal, wf, gamma) * geo.length[:, None]
    return jax.ops.segment_sum(-f, a, n) + jax.ops.segment_sum(f, b, n)


def step(state: State, dt, cfg: Config) -> State:
    pos, q = state
    topo1 = mesh.topology(pos, cfg.box)
    geo1 = mesh.geometry(pos, topo1, cfg.box)
    vel = seed_velocity(pos, primitives(q, geo1, cfg.gamma), geo1, cfg)
    r1 = rhs(pos, q, vel, topo1, cfg)
    pos2 = jnp.mod(pos + vel * dt, cfg.box)
    topo2 = mesh.topology(pos2, cfg.box)
    r2 = rhs(pos2, q + dt * r1, vel, topo2, cfg)
    return State(pos2, q + 0.5 * dt * (r1 + r2))


def stable_dt(state: State, cfg: Config, cfl=0.3):
    """Largest stable step for the current state, as a plain number (no gradient)."""
    pos, q = jax.lax.stop_gradient(state)
    geo = mesh.geometry(pos, mesh.topology(pos, cfg.box), cfg.box)
    w = primitives(q, geo, cfg.gamma)
    cs = hydro.sound_speed(w, cfg.gamma)
    radius = jnp.sqrt(geo.area / jnp.pi)
    signal = cs * (1.0 + cfg.reg_speed) if cfg.moving else cs + jnp.linalg.norm(w[:, 1:3], axis=1)
    return float(cfl * jnp.min(radius / signal))


def run(state: State, dt, n_steps: int, cfg: Config, segment: int = 0) -> State:
    """`n_steps` steps of size `dt`. Reverse mode stores one state per step; with `segment`
    > 0 it stores one per segment of that many steps and recomputes inside each."""
    one = jax.checkpoint(lambda s: step(s, dt, cfg))

    def body(s, _):
        return one(s), None

    if segment <= 0 or n_steps % segment:
        return jax.lax.scan(body, state, length=n_steps)[0]

    @jax.checkpoint
    def chunk(s, _):
        return jax.lax.scan(body, s, length=segment)[0], None

    return jax.lax.scan(chunk, state, length=n_steps // segment)[0]

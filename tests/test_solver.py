import jax
import jax.numpy as jnp
import numpy as np

import dvf  # noqa: F401
from dvf import mesh, problems, solver

BOX = 4.0
CFG = solver.Config(box=BOX)
POS = problems.lattice(12, BOX, jitter=0.3, seed=1)


def wave(theta):
    def init(c):
        one = jnp.ones(c.shape[0])
        rho = 1.0 + theta[0] * jnp.sin(2 * jnp.pi * c[:, 0] / BOX)
        p = 1.0 + 0.2 * jnp.cos(2 * jnp.pi * c[:, 1] / BOX)
        return jnp.stack([rho, theta[1] * one, 0.2 * one, p], axis=1)

    return init


THETA = jnp.array([0.3, 0.5])
DT = solver.stable_dt(solver.initial_state(POS, wave(THETA), CFG), CFG)


def loss(theta, n_steps=20, segment=0):
    s = solver.run(solver.initial_state(POS, wave(theta), CFG), DT, n_steps, CFG, segment)
    geo = mesh.geometry(s.pos, mesh.topology(s.pos, BOX), BOX)
    rho = s.q[:, 0] / geo.area
    return jnp.sum(geo.area * rho**2 * jnp.sin(geo.centroid[:, 0]))


def test_run_conserves_mass_momentum_and_energy():
    s0 = solver.initial_state(POS, wave(THETA), CFG)
    s1 = jax.jit(lambda s: solver.run(s, DT, 30, CFG))(s0)
    total = lambda s: np.asarray(s.q.sum(axis=0))
    np.testing.assert_allclose(total(s1), total(s0), rtol=1e-12, atol=1e-12)


def test_uniform_flow_stays_uniform_on_a_moving_mesh():
    w0 = jnp.array([1.0, 0.7, -0.3, 0.5])
    s0 = solver.initial_state(POS, lambda c: jnp.tile(w0, (c.shape[0], 1)), CFG)
    s1 = solver.run(s0, DT, 20, CFG)
    geo = mesh.geometry(s1.pos, mesh.topology(s1.pos, BOX), BOX)
    w = solver.primitives(s1.q, geo, CFG.gamma)
    assert float(jnp.abs(w - w0).max()) < 1e-3


def test_reverse_mode_matches_finite_differences():
    g = jax.jit(jax.grad(loss))(THETA)
    f = jax.jit(loss)
    for i in range(2):
        e = jnp.zeros(2).at[i].set(1e-5)
        fd = (f(THETA + e) - f(THETA - e)) / 2e-5
        assert abs(float(fd - g[i])) < 1e-7 * abs(float(fd)), (i, float(fd), float(g[i]))


def test_forward_and_reverse_mode_agree():
    v = jnp.array([0.7, -0.4])
    _, jv = jax.jvp(loss, (THETA,), (v,))
    g = jax.grad(loss)(THETA)
    assert abs(float(jv - g @ v)) < 1e-10 * abs(float(jv))


def test_segmented_checkpointing_gives_the_same_gradient():
    g1 = jax.grad(loss)(THETA, 20, 0)
    g2 = jax.grad(loss)(THETA, 20, 5)
    np.testing.assert_allclose(np.asarray(g2), np.asarray(g1), rtol=1e-12)


def test_gradient_at_rest_has_the_right_size():
    # The starting point of the logo problem. The loss has a kink at rest, where the limiter
    # switches on for any perturbation, so compare a one-sided slope along the gradient
    # rather than central differences. On an exact lattice the tangent dynamics at rest are
    # unstable and this gradient comes out 1e8 times too large; the seeds are jittered.
    from dvf import hydro

    pos = problems.lattice(12, BOX, jitter=0.3, seed=2)
    area = mesh.geometry(pos, mesh.topology(pos, BOX), BOX).area
    n = pos.shape[0]

    def final_misfit(v):
        w = jnp.concatenate([jnp.ones((n, 1)), v.reshape(n, 2), jnp.full((n, 1), 0.1)], axis=1)
        s = solver.State(pos, hydro.prim_to_cons(w, CFG.gamma) * area[:, None])
        s = solver.run(s, 0.02, 40, CFG)
        geo = mesh.geometry(s.pos, mesh.topology(s.pos, BOX), BOX)
        rho = s.q[:, 0] / geo.area
        return jnp.sum(geo.area * (rho - 1.0 - 0.3 * jnp.sin(geo.centroid[:, 0])) ** 2)

    v0 = jnp.zeros(2 * n)
    g = jax.grad(final_misfit)(v0)
    norm = float(jnp.linalg.norm(g))
    eps = 1e-6
    slope = float(final_misfit(-eps * g / norm) - final_misfit(v0)) / eps
    assert 0.5 < -slope / norm < 2.0, (slope, norm)

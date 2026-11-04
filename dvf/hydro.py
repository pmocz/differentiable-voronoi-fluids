"""Ideal-gas Euler: state conversion and the HLLC flux through a moving face.

Every branch is evaluated and the right one selected, so a branch that is not taken must
still produce finite numbers: an infinity there would turn the gradient into NaN through
`jnp.where`. Denominators are kept away from zero for that reason.
"""

import jax.numpy as jnp


def safe(x, eps=1e-100):
    """`x`, kept at least `eps` from zero, for denominators in branches that may not be
    taken. The derivative of 1 / x goes as 1 / x^2, which has to stay finite as well."""
    return jnp.where(jnp.abs(x) < eps, jnp.where(x < 0.0, -eps, eps), x)


def prim_to_cons(w, gamma):
    """[rho, vx, vy, p] to conserved densities [rho, rho vx, rho vy, E]."""
    rho, vx, vy, p = w[..., 0], w[..., 1], w[..., 2], w[..., 3]
    e = p / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy)
    return jnp.stack([rho, rho * vx, rho * vy, e], axis=-1)


def cons_to_prim(u, gamma):
    rho = u[..., 0]
    vx, vy = u[..., 1] / rho, u[..., 2] / rho
    p = (gamma - 1.0) * (u[..., 3] - 0.5 * rho * (vx * vx + vy * vy))
    return jnp.stack([rho, vx, vy, p], axis=-1)


def sound_speed(w, gamma):
    return jnp.sqrt(gamma * w[..., 3] / w[..., 0])


def hllc(left, right, gamma):
    """HLLC flux for states [rho, u_normal, u_tangential, p], as [mass, normal momentum,
    tangential momentum, energy]."""
    rl, ul, vl, pl = (left[..., k] for k in range(4))
    rr, ur, vr, pr = (right[..., k] for k in range(4))
    cl = jnp.sqrt(gamma * pl / rl)
    cr = jnp.sqrt(gamma * pr / rr)
    el = pl / (gamma - 1.0) + 0.5 * rl * (ul * ul + vl * vl)
    er = pr / (gamma - 1.0) + 0.5 * rr * (ur * ur + vr * vr)

    pstar = jnp.maximum(0.5 * (pl + pr) - 0.125 * (ur - ul) * (rl + rr) * (cl + cr), 0.0)

    def shock_factor(pk):
        ratio = jnp.maximum(pstar / pk - 1.0, 0.0)
        return jnp.sqrt(1.0 + (gamma + 1.0) / (2.0 * gamma) * ratio)

    sl = jnp.minimum(ul - cl * shock_factor(pl), ur - cr)
    sr = jnp.maximum(ur + cr * shock_factor(pr), ul + cl)
    sstar = (pr - pl + rl * ul * (sl - ul) - rr * ur * (sr - ur)) / safe(
        rl * (sl - ul) - rr * (sr - ur)
    )

    def flux(rho, u, v, p, e):
        return jnp.stack([rho * u, rho * u * u + p, rho * u * v, u * (e + p)], axis=-1)

    def side(rho, u, v, p, e, s):
        f = flux(rho, u, v, p, e)
        c = rho * (s - u) / safe(s - sstar)
        star = jnp.stack(
            [c, c * sstar, c * v, c * (e / rho + (sstar - u) * (sstar + p / safe(rho * (s - u))))],
            axis=-1,
        )
        u0 = jnp.stack([rho, rho * u, rho * v, e], axis=-1)
        return f + s[..., None] * (star - u0)

    fl = flux(rl, ul, vl, pl, el)
    fr = flux(rr, ur, vr, pr, er)
    fls = side(rl, ul, vl, pl, el, sl)
    frs = side(rr, ur, vr, pr, er, sr)
    return jnp.where(
        (sl >= 0.0)[..., None],
        fl,
        jnp.where((sstar >= 0.0)[..., None], fls, jnp.where((sr >= 0.0)[..., None], frs, fr)),
    )


def face_flux(wl, wr, normal, wf, gamma):
    """Lab-frame flux per unit length through faces with unit `normal` (left to right)
    moving with velocity `wf`."""
    t = jnp.stack([-normal[..., 1], normal[..., 0]], axis=-1)

    def rotate(w):
        rel = w[..., 1:3] - wf
        return jnp.stack(
            [w[..., 0], jnp.sum(rel * normal, axis=-1), jnp.sum(rel * t, axis=-1), w[..., 3]],
            axis=-1,
        )

    f = hllc(rotate(wl), rotate(wr), gamma)
    fm = f[..., 0]
    mom = normal * f[..., 1:2] + t * f[..., 2:3]
    fe = f[..., 3] + jnp.sum(mom * wf, axis=-1) + 0.5 * jnp.sum(wf * wf, axis=-1) * fm
    mom_lab = mom + fm[..., None] * wf
    return jnp.stack([fm, mom_lab[..., 0], mom_lab[..., 1], fe], axis=-1)

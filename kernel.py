import time

import jax
from jax import numpy as jnp


class Kernel:
    @staticmethod
    @jax.jit
    def _get_s(s, rng_key):
        new_rng_key, subkey = jax.random.split(rng_key)
        xi = jax.random.uniform(subkey, shape=s.shape, dtype=s.dtype)
        return -jnp.log(xi), new_rng_key

    @staticmethod
    @jax.jit
    def _move(ph_xyz, ph_u_xyz, ph_s, cl_z0, cl_z1, cl_mu_t):
        ph_uz = ph_u_xyz[..., 2]

        inv_uz = jnp.where(ph_uz != 0, 1.0 / ph_uz, jnp.inf)
        db = jnp.where(
            ph_uz < 0,
            (cl_z0 - ph_xyz[..., 2]) * inv_uz,
            (cl_z1 - ph_xyz[..., 2]) * inv_uz
        )

        hit_bound = db * cl_mu_t <= ph_s
        new_ph_xyz = jnp.where(
            hit_bound[:, None],
            ph_xyz + ph_u_xyz * db[:, None],
            ph_xyz + ph_u_xyz * ph_s[:, None] / cl_mu_t[:, None]
        )
        new_ph_s = jnp.where(
            hit_bound,
            ph_s - db * cl_mu_t,
            0
        )

        return new_ph_xyz, new_ph_s

    @staticmethod
    @jax.jit
    def _absorb(ph_w, cl_mu_a, cl_mu_t):
        mu_ratio = cl_mu_a / cl_mu_t
        delta_w = ph_w * mu_ratio
        new_ph_w = ph_w - delta_w
        return new_ph_w, delta_w

    @staticmethod
    @jax.jit
    def _scatter(ph_u_xyz, cl_g, rng_key):
        new_rng_key, subkey = jax.random.split(rng_key)
        xi = jax.random.uniform(subkey, shape=cl_g.shape, dtype=cl_g.dtype)

        cosine = jnp.where(
            cl_g == 0,
            2 * xi - 1,
            (1 + cl_g ** 2 - ((1 - cl_g ** 2) / (1 - cl_g + 2 * cl_g * xi)) ** 2) / (2 * cl_g)
        )
        theta = jnp.arccos(cosine)

        new_rng_key, subkey = jax.random.split(new_rng_key)
        xi = jax.random.uniform(subkey, shape=cl_g.shape, dtype=cl_g.dtype)
        phi = 2 * jnp.pi * xi

        less_than_one = jnp.abs(ph_u_xyz[..., 2]) < 0.99999
        ux, uy, uz = ph_u_xyz[..., 0], ph_u_xyz[..., 1], ph_u_xyz[..., 2]

        new_ph_u_xyz = ph_u_xyz.at[..., 0].set(jnp.where(
            less_than_one,
            jnp.sin(theta) * (ux * uz * jnp.cos(phi) - uy * jnp.sin(phi)) / jnp.sqrt(1 - uz ** 2) + ux * cosine,
            jnp.sin(theta) * jnp.cos(phi)
        ))
        new_ph_u_xyz = new_ph_u_xyz.at[..., 1].set(jnp.where(
            less_than_one,
            jnp.sin(theta) * (uy * uz * jnp.cos(phi) + ux * jnp.sin(phi)) / jnp.sqrt(1 - uz ** 2) + uy * cosine,
            jnp.sin(theta) * jnp.sin(phi)
        ))
        new_ph_u_xyz = new_ph_u_xyz.at[..., 2].set(jnp.where(
            less_than_one,
            -jnp.sin(theta) * jnp.cos(phi) * jnp.sqrt(1 - uz ** 2) + uz * cosine,
            jnp.sign(uz) * cosine
        ))

        return new_ph_u_xyz, new_rng_key

    @staticmethod
    @jax.jit
    def _reflect_transmit(ph_u_xyz, ph_cl, ly_n, rng_key):
        n_i = ly_n[ph_cl]
        a_i = jnp.arccos(jnp.abs(ph_u_xyz[..., 2]))

        less_than_zero = ph_u_xyz[..., 2] < 0
        n_t = jnp.where(
            less_than_zero,
            ly_n[ph_cl - 1],
            ly_n[ph_cl + 1]
        )
        direct = jnp.where(
            less_than_zero,
            -1, 1
        )

        con = jnp.logical_and(n_i > n_t, a_i > jnp.arcsin(n_t / n_i))
        a_t = jnp.where(
            con,
            jnp.pi / 2,
            jnp.arcsin(n_i * jnp.sin(a_i) / n_t)
        )
        r = jnp.where(
            con,
            1,
            ((jnp.sin(a_i - a_t) / jnp.sin(a_i + a_t)) ** 2 + (jnp.tan(a_i - a_t) / jnp.tan(a_i + a_t)) ** 2) * 0.5
        )

        new_rng_key, subkey = jax.random.split(rng_key)
        xi = jax.random.uniform(subkey, shape=ph_cl.shape)

        xi_less_than_r = xi <= r
        new_uz = jnp.where(xi_less_than_r, -ph_u_xyz[..., 2], ph_u_xyz[..., 2])
        new_cl = jnp.where(
            xi_less_than_r,
            ph_cl,
            ph_cl + direct
        )

        new_dead = jnp.logical_or(new_cl == 0, new_cl == ly_n.shape[-1] - 1)

        con3 = jnp.logical_or(xi_less_than_r, new_dead)

        new_ph_u_xyz = ph_u_xyz.at[..., 0:1].set(jnp.where(
            con3[:, None],
            ph_u_xyz[..., 0:1],
            ph_u_xyz[..., 0:1] * n_i[:, None] / n_t[:, None]
        ))
        new_ph_u_xyz = new_ph_u_xyz.at[..., 2].set(jnp.where(
            con3,
            new_uz,
            jnp.sign(new_uz) * jnp.cos(a_t)
        ))

        return new_ph_u_xyz, new_cl, new_dead, new_rng_key

    @staticmethod
    @jax.jit
    def _terminate(w, m, rng_key):
        new_rng_key, subkey = jax.random.split(rng_key)
        xi = jax.random.uniform(subkey, shape=w.shape, dtype=jnp.float16)

        less_than_one_over_m = xi <= 1 / m

        new_w = jnp.where(less_than_one_over_m, m * w, 0)
        new_dead = ~less_than_one_over_m
        return new_w, new_dead, new_rng_key

    @staticmethod
    @jax.jit
    def _update_grid(dw, ph_xyz, ph_sct, gd_dr, gd_dz, gd_nr, gd_nz):
        ir = (jnp.sqrt(ph_xyz[..., 0] ** 2 + ph_xyz[..., 1] ** 2) / gd_dr)
        iz = (ph_xyz[..., 2] / gd_dz).astype(jnp.int16)
        in_grid = jnp.logical_and(ph_sct != 0, jnp.logical_and(ir < gd_nr, iz < gd_nz))

        i_rz = jnp.zeros((*ph_xyz.shape[:-1], 2), dtype=jnp.int16)

        i_rz = i_rz.at[..., 0].set(jnp.where(
            in_grid,
            ir.astype(jnp.int16),
            0
        ))
        i_rz = i_rz.at[..., 1].set(jnp.where(
            in_grid,
            iz,
            0
        ))

        new_dw = jnp.where(
            in_grid,
            dw / ((ir * gd_dr + 1) ** 2),
            0
        )

        return i_rz, new_dw


    @staticmethod
    @jax.jit
    def photon_forward(
        ph_xyz, ph_u_xyz, ph_s, ph_cl, ph_w, ph_sct, ph_dead,
        ly_z0, ly_z1, ly_n, ly_g, ly_mu_a, ly_mu_t,
        gd_dr, gd_dz, gd_nr, gd_nz, gd_i_rz,
        w, m, rng_key
    ):
        # get s
        _ph_s, new_rng_key = Kernel._get_s(ph_s, rng_key)
        new_ph_s = jnp.where(
            ph_s == 0,
            _ph_s,
            ph_s
        )

        # move
        new_ph_xyz, new_ph_s = Kernel._move(ph_xyz, ph_u_xyz, new_ph_s, ly_z0[ph_cl], ly_z1[ph_cl], ly_mu_t[ph_cl])

        s_is_zero_2 = new_ph_s == 0

        # absorb
        _ph_w, d_w = Kernel._absorb(ph_w, ly_mu_a[ph_cl], ly_mu_t[ph_cl])
        new_ph_w = jnp.where(
            s_is_zero_2,
            _ph_w,
            ph_w
        )

        # out_grid
        i_rz, _new_dw = Kernel._update_grid(d_w, ph_xyz, ph_sct, gd_dr, gd_dz, gd_nr, gd_nz)
        new_dw = jnp.where(
            jnp.logical_and(s_is_zero_2, jnp.logical_not(ph_dead)),
            _new_dw,
            0
        )
        ir, iz = i_rz[..., 0], i_rz[..., 1]
        new_gd_i_rz = gd_i_rz.at[iz, ir].add(new_dw)

        # scatter
        _ph_u_xyz, new_rng_key = Kernel._scatter(ph_u_xyz, ly_g[ph_cl], new_rng_key)
        new_ph_sct = jnp.where(
            s_is_zero_2,
            ph_sct + 1,
            ph_sct
        )
        # reflect_transmit
        _ph_u_xyz_2, _ph_cl, _ph_dead, new_rng_key = Kernel._reflect_transmit(ph_u_xyz, ph_cl, ly_n, new_rng_key)

        new_ph_u_xyz = jnp.where(
            s_is_zero_2[:, None],
            _ph_u_xyz,
            _ph_u_xyz_2
        )
        new_ph_cl = jnp.where(
            s_is_zero_2,
            ph_cl,
            _ph_cl
        )
        new_ph_dead = jnp.where(
            s_is_zero_2,
            False,
            _ph_dead
        )

        not_dead_low_weight = jnp.logical_and(jnp.logical_not(new_ph_dead), new_ph_w < w)
        _ph_w_2, _ph_dead, new_rng_key = Kernel._terminate(ph_w, m, rng_key)

        new_new_ph_dead = jnp.logical_or(new_ph_dead, jnp.logical_and(not_dead_low_weight, _ph_dead))
        new_new_ph_w = jnp.where(
            new_new_ph_dead,
            new_ph_w,
            _ph_w_2
        )

        return (
            new_ph_xyz, new_ph_u_xyz, new_ph_s, new_ph_cl, new_new_ph_w, new_new_ph_dead, new_ph_sct,
            new_gd_i_rz,
            new_rng_key
        )

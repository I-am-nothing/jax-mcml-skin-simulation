import numpy as np
import jax.numpy as jnp
import jax

class Photons:
    def __init__(self, photons_num, r_sp, config):
        self.photons_num = photons_num
        self.float_dtype = jnp.float32 if config["fp_16"] is None or config["fp_16"] == False else jnp.float16

        self.w = jnp.ones(self.photons_num, dtype=self.float_dtype) - r_sp

        self.xyz = jnp.zeros((self.photons_num, 3), dtype=self.float_dtype)
        self.s = jnp.zeros(self.photons_num, dtype=self.float_dtype)
        self.cl = jnp.ones(self.photons_num, dtype=jnp.int8)

        self.sct = jnp.zeros(self.photons_num, dtype=jnp.int16)

        u_xyz = jnp.zeros((self.photons_num, 3), dtype=self.float_dtype)
        self.u_xyz = u_xyz.at[..., 2].set(1)

        self.dead = jnp.zeros(self.photons_num, dtype=jnp.bool)

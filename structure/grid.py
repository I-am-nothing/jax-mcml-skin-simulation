import jax.numpy as jnp

class Grid:
    def __init__(self, config):
        self.float_dtype = jnp.float32 if config["fp_16"] is None or config["fp_16"] == False else jnp.float16
        self.dr, self.dz = config["grid_dr_dz"]
        self.nr, self.nz = config["grid_nr_nz"]
        self.i_rz = jnp.zeros((self.nz, self.nr), dtype=self.float_dtype)
        self.zz = (self.nz - 1) * self.dz

    def get_output(self, layer):
        cl = 1
        output = jnp.zeros_like(self.i_rz)
        zt = self.zz
        z_first = (layer.n[0] / self.dz).astype(jnp.int16)

        while zt > 0 and cl < layer.shape[0]-1:
            thickness = layer.z1[cl] - layer.z0[cl]
            iz0 = (layer.z0[cl] / self.dz).astype(jnp.int16) - z_first

            if zt > thickness:
                iz1 = (layer.z1[cl] / self.dz).astype(jnp.int16) - z_first
            else:
                iz1 = -1

            if iz1 == iz0:
                iz1 += 1
            output = output.at[iz0:iz1].set(self.i_rz[iz0:iz1] / layer.mu_a[cl])
            cl += 1
            zt -= thickness
        # print("\n", output.max())
        return output
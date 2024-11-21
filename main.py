import json
import os.path
import time
import jax.numpy as jnp
import jax
import numpy as np

import utils
import scipy
from matplotlib import pyplot as plt
from dataset.skin_bom import SkinBomDataset
from kernel import Kernel
from structure.grid import Grid
from structure.photons import Photons


def plot(output):
    response = scipy.io.loadmat("data/custom_light_response.mat")['Resp_func'][::2]
    output = jnp.transpose(output, (1, 2, 0))
    bgr = jnp.dot(output, response)
    bgr = bgr / jnp.max(bgr)
    rgb = bgr[..., ::-1]

    plt.figure()
    plt.title("Figure rz")
    plt.imshow(rgb)
    plt.show()


if __name__ == "__main__":
    with open("data/config.json") as f:
        config = json.load(f)
    dataset = SkinBomDataset(config)

    ph_num = config["number_of_photons"]
    w, m = config["weight"], config["m_weight"]
    gd_nr, gd_nz = config["grid_nr_nz"]
    shuffle = config["shuffle"]
    save_path = config["save_path"]

    if not os.path.isdir(f"{save_path}/bom"):
        os.makedirs(f"{save_path}/bom")
    if not os.path.isdir(f"{save_path}/aop"):
        os.makedirs(f"{save_path}/aop")

    index = list(range(len(dataset)))
    if shuffle:
        np.random.shuffle(index)

    rng_key = jax.random.PRNGKey(int(time.time() * 10000 % 10000))

    for j in range(len(dataset)):
        idx, layers, bom = dataset[index[j]]
        output = jnp.zeros((layers.shape[0], gd_nz, gd_nr))
        layers = utils.to_jax(layers)

        for i in range(layers.shape[0]):
            layer = layers[i]
            r_sp = ((layer.n[0] - layer.n[1]) / (layer.n[0] + layer.n[1])) ** 2
            photons = Photons(ph_num, r_sp, config)

            grid = Grid(config)
            while photons.dead.sum() < ph_num:
                photons.xyz, photons.u_xyz, photons.s, photons.cl, photons.w, new_ph_dead, photons.sct, grid.i_rz, rng_key = Kernel.photon_forward(
                    photons.xyz, photons.u_xyz, photons.s, photons.cl, photons.w, photons.sct, photons.dead,
                    layer.z0, layer.z1, layer.n, layer.g, layer.mu_a, layer.mu_t,
                    grid.dr, grid.dz, grid.nr, grid.nz, grid.i_rz,
                    w, m, rng_key
                )
                photons.dead = jnp.logical_or(photons.dead, new_ph_dead)

            xdd = grid.get_output(layer)
            output = output.at[i].set(xdd)

            print(f"\rfinish layer: {i+1}/{layers.shape[0]} on idx: {idx}", end="")

        output = np.array(output)

        bom.save(f"{save_path}/bom/{idx}_bom.npy")
        np.save(f"{save_path}/aop/{idx}_aop.npy", output)
	
        # print()
        # plot(output)

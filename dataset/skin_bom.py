import time

import numpy as np
import utils

from structure.bom import BOM
from dataset.permutation import PermutationDataset
from structure.layers import Layer
from structure.skin_lut import SkinLUT


class SkinBomDataset(PermutationDataset):
    def __init__(self, config):
        super().__init__(config)
        self.float_dtype = np.float32 if config["fp_16"] is None or config["fp_16"] == False else np.float16
        self.spectrum = config["spectrum"]
        self.n_photons = config["number_of_photons"]
        self.n_layers = config["num_of_iop_layers_without_ambient"]

        photon_lut = utils.load_yaml(config["photon_lut_path"])["LAMBDA_PERCENTAGE"]
        wavelengths = np.array(list(photon_lut.keys()), dtype=self.float_dtype)
        values = np.array(list(photon_lut.values()), dtype=self.float_dtype)
        target_wavelengths = np.linspace(self.spectrum[0], self.spectrum[1], self.spectrum[2])

        self.photon_lut = utils.linear_interpolation(values, wavelengths, target_wavelengths)
        self.skin_lut = SkinLUT.init(config)


    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        idx, params = super().__getitem__(index)
        params = BOM(params.shape[0], params)

        layers = Layer.init((self.spectrum[2], self.n_layers+2), float_dtype=self.float_dtype)

        layers.mu_a[:, 1:-1] = self._compute_mua(params)
        layers.mu_s[:, 1:-1] = self._compute_mus(params)
        layers.mu_t = layers.mu_a + layers.mu_s
        layers.g[:, 1:-1] = params.g
        layers.n[:, 1:-1] = params.n

        param_d = np.cumsum(np.concatenate(([1.], params.d)))
        layers.z0[:, 1:-1] = param_d[:-1]
        layers.z1[:, 1:-1] = param_d[1:]
        layers.z1[:, -1] = np.inf

        return idx, layers, params

    def _compute_mua(self, params):
        x = np.outer(self.skin_lut.oxy, params.b * params.s)
        x = x + np.outer(self.skin_lut.deoxy, params.b * (1 - params.s))
        x = x + np.outer(self.skin_lut.water, params.w)
        x = x + np.outer(self.skin_lut.fat, params.f)
        x = x + np.outer(self.skin_lut.mel, params.m)

        return x

    def _compute_mus(self, params):
        l = self.photon_lut * self.n_photons
        x = 2.0E5 * np.power(l, -1.5) + 2.0E12 * np.power(l, -4)
        x = np.outer(x, 1 / (1 - params.g))

        return x

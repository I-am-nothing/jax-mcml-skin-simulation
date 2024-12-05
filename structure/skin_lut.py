import numpy as np
from tensorstore import dtype

import utils


class SkinLUT:
    @staticmethod
    def init(config):
        skin_lut = utils.load_yaml(config["skin_lut_path"])
        spectrum = config["spectrum"]
        float_dtype = np.float32 if config["fp_16"] is None or config["fp_16"] == False else np.float16

        for i in skin_lut.keys():
            values = np.array(list(skin_lut[i].values()), dtype=float_dtype)
            wavelengths = np.array(list(skin_lut[i].keys()), dtype=float_dtype)
            target_wavelengths = np.linspace(spectrum[0], spectrum[1], spectrum[2])
            skin_lut[i] = utils.linear_interpolation(values, wavelengths, target_wavelengths, dtype=dtype)

        keys = list(skin_lut.keys())
        data = np.stack([skin_lut[k] for k in keys])
        return SkinLUT(keys, data)

    def __init__(self, keys, data):
        self._keys = keys
        self._data = data

    def to(self, device):
        self._data = self._data.to(device)

    def keys(self, dim):
        return self._data.keys()

    @property
    def oxy(self):
        return self._data[0]

    @oxy.setter
    def oxy(self, values):
        self._data[0] = values

    @property
    def deoxy(self):
        return self._data[1]

    @deoxy.setter
    def deoxy(self, values):
        self._data[1] = values

    @property
    def water(self):
        return self._data[2]

    @water.setter
    def water(self, values):
        self._data[2] = values

    @property
    def fat(self):
        return self._data[3]

    @fat.setter
    def fat(self, values):
        self._data[3] = values

    @property
    def mel(self):
        return self._data[3]

    @mel.setter
    def mel(self, values):
        self._data[3] = values


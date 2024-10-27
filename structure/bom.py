import numpy as np

class BOM:
    def __init__(self, size, data):
        # data columns: 0-s, 1-b, 2-w, 3-f, 4-m, 5-g, 6-d, 7-n
        self._size = size
        self._data = data

    @property
    def shape(self):
        return list(self._size)

    def size(self, dim):
        return self._size[dim]

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)

        new_shape = self._data[idx].shape
        return BOM(new_shape[:-1], self._data[idx])

    def __setitem__(self, idx, value):
        if not isinstance(idx, tuple):
            idx = (idx,)

        self._data[idx + (..., 0)] = value.s
        self._data[idx + (..., 1)] = value.b
        self._data[idx + (..., 2)] = value.w
        self._data[idx + (..., 3)] = value.f
        self._data[idx + (..., 4)] = value.m
        self._data[idx + (..., 5)] = value.g
        self._data[idx + (..., 6)] = value.d
        self._data[idx + (..., 7)] = value.n

    def save(self, path):
        np.save(path, self._data.reshape(-1))

    @property
    def s(self):
        return self._data[..., 0]

    @s.setter
    def s(self, values):
        self._data[..., 0] = values

    @property
    def b(self):
        return self._data[..., 1]

    @b.setter
    def b(self, values):
        self._data[..., 1] = values

    @property
    def w(self):
        return self._data[..., 2]

    @w.setter
    def w(self, values):
        self._data[..., 2] = values

    @property
    def f(self):
        return self._data[..., 3]

    @f.setter
    def f(self, values):
        self._data[..., 3] = values

    @property
    def m(self):
        return self._data[..., 4]

    @m.setter
    def m(self, values):
        self._data[..., 4] = values

    @property
    def g(self):
        return self._data[..., 5]

    @g.setter
    def g(self, values):
        self._data[..., 5] = values

    @property
    def d(self):
        return self._data[..., 6]

    @d.setter
    def d(self, values):
        self._data[..., 6] = values

    @property
    def n(self):
        return self._data[..., 7]

    @n.setter
    def n(self, values):
        self._data[..., 7] = values
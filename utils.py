
import numpy as np
import jax.numpy as jnp

import yaml
from scipy import interpolate


def linear_interpolation(x, a, b, dtype=np.float32):
    if np.max(b) > np.max(a):
        a = np.concatenate((a, [np.max(b)]))
        x = np.concatenate((x, [0]))
    if np.min(b) < np.min(a):
        a = np.concatenate(([np.min(b)], a))
        x = np.concatenate(([0], x))

    # Perform linear interpolation
    linear_interp = interpolate.interp1d(a, x, kind='linear')
    target_x = linear_interp(b)

    return np.array(target_x, dtype=dtype)

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

def to_jax(obj):
    for attr_name, attr_value in obj.__dict__.items():
        if isinstance(attr_value, np.ndarray):
            obj.__dict__[attr_name] = jnp.array(attr_value)

    return obj

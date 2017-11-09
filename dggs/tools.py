import numpy as np


def apply_affine(A, x, y):
    assert x.shape == y.shape
    assert hasattr(A, '__mul__')

    sh = x.shape

    x, y = A*np.vstack([x.ravel(), y.ravel()])
    return x.reshape(sh), y.reshape(sh)

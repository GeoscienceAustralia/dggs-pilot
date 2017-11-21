import numpy as np


def polygon_path(x, y=None):
    """A little bit like numpy.meshgrid, except returns only boundary values and
    limited to 2d case only.

    Examples:
      [0,1], [3,4] =>
      array([[0, 1, 1, 0, 0],
             [3, 3, 4, 4, 3]])

      [0,1] =>
      array([[0, 1, 1, 0, 0],
             [0, 0, 1, 1, 0]])
    """
    if y is None:
        y = x

    return np.vstack([
        np.vstack([x, np.full_like(x, y[0])]).T,
        np.vstack([np.full_like(y, x[-1]), y]).T[1:],
        np.vstack([x, np.full_like(x, y[-1])]).T[::-1][1:],
        np.vstack([np.full_like(y, x[0]), y]).T[::-1][1:]]).T


def nodata_mask(a, nodata=None):
    """ Returns boolean array with True where there is no valid data
    """
    if a.dtype.kind == 'f':
        if nodata is None or np.isnan(nodata):
            return np.isnan(a)
        else:
            return a == nodata
    else:
        if nodata is None:
            # Integer image without nodata field is always valid everywhere
            return np.zeros_like(a, dtype='bool')
        else:
            return a == nodata


def valid_data_mask(a, nodata=None):
    """ Returns boolean array with True where there is valid data
    """
    if a.dtype.kind == 'f':
        if nodata is None or np.isnan(nodata):
            return ~np.isnan(a)
        else:
            return a != nodata
    else:
        if nodata is None:
            # Integer image without nodata field is always valid everywhere
            return np.ones_like(a, dtype='bool')
        else:
            return a != nodata


def nodata_to_num(a, nodata=None):
    """ Like `numpy.nan_to_num` but supporting nodata values

    Replace nodata values with zeros
    """
    if a.dtype.kind == 'f':
        a = np.nan_to_num(a)
        if nodata is not None and not np.isnan(nodata):
            a[a == nodata] = 0
    else:
        if nodata is not None:
            a = a.copy()
            a[a == nodata] = 0
        else:
            pass  # Note output is same data as input, no copy

    return a


def sum3x3(a, dtype=None):
    assert a.shape[0] % 3 == 0 and a.shape[1] % 3 == 0

    if dtype is None:
        dtype = dict(f='float32',
                     i='int64',
                     b='uint64',  # boolean -> uint64
                     u='uint64').get(a.dtype.kind, a.dtype)

    aa = a[0::3].astype(dtype) + a[1::3].astype(dtype) + a[2::3].astype(dtype)
    return aa[:, 0::3] + aa[:, 1::3] + aa[:, 2::3]


def logical_and_3x3(a):
    """ Combine 3x3 pixel with logical AND into one output pixel
    """
    assert a.dtype == np.bool
    assert a.shape[0] % 3 == 0 and a.shape[1] % 3 == 0
    aa = a[0::3, :] * a[1::3, :] * a[2::3, :]
    return aa[:, 0::3] * aa[:, 1::3] * aa[:, 2::3]


def logical_or_3x3(a):
    """ Combine 3x3 pixel with logical OR into one output pixel
    """
    assert a.dtype == np.bool
    assert a.shape[0] % 3 == 0 and a.shape[1] % 3 == 0
    return sum3x3(a, dtype=np.bool)


def expand_3x3(a):
    """ Convert every pixel into 3x3 block
    """
    (h, w) = a.shape[:2]
    new_shape = (h*3, w*3) + a.shape[2:]
    o = np.empty(new_shape, dtype=a.dtype)

    for i in range(3):
        for j in range(3):
            o[i::3, j::3] = a

    return o


def apply_affine(A, x, y):
    assert x.shape == y.shape
    assert hasattr(A, '__mul__')

    sh = x.shape

    x, y = A*np.vstack([x.ravel(), y.ravel()])
    return x.reshape(sh), y.reshape(sh)


def geo_boundary(affine, shape):
    h, w = shape
    x = np.linspace(0, w, w+1)
    y = np.linspace(0, h, h+1)
    x, y = polygon_path(x, y)
    return apply_affine(affine, x, y)

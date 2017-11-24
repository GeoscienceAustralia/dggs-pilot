import numpy as np
import matplotlib.pyplot as plt

from .tools import polygon_path
from . import DGGS

dg = DGGS()


def _get_ax(ax):
    return ax if ax else plt.gca()


def is_rgba(ds):
    return set('blue green red alpha'.split()).issubset(set(v for v in ds.data_vars))


def is_rgb(ds):
    return set('blue green red'.split()).issubset(set(v for v in ds.data_vars))


def as_rgb(ds):
    return np.dstack([ds[v] for v in 'red green blue'.split()])


def as_rgba(ds):
    return np.dstack([ds[v] for v in 'red green blue alpha'.split()])


def cell_bounds(addr):
    tr, *_ = dg.pixel_coord_transform(addr, 1, 1, native=True, no_offset=True)
    x_min, y_min = tr(0, 0)
    x_max, y_max = tr(1, 1)
    return (x_min, x_max, y_min, y_max)


def cell_center(addr):
    tr, *_ = dg.pixel_coord_transform(addr, native=True)
    return tr(0, 0)


def merge_extents(e1, e2):
    if e1 is None:
        return e2

    return [min(e1[0], e2[0]), max(e1[1], e2[1]),
            min(e1[2], e2[2]), max(e1[3], e2[3])]


def hide_axis(ax):
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.axis('off')


def plot_bbox(extents, style='-', ax=None, **kwargs):
    """
    extents : (left, right, bottom, top)
    """

    ax = _get_ax(ax)
    extents = [float(v) for v in extents]
    return ax.plot(*polygon_path(extents[:2], extents[2:]), style, **kwargs)


def add_cell_plot(cell_addr, style='-', ax=None,  w=1, h=1, crs=None, native=False, **kwargs):
    tr, (maxW, maxH) = dg.pixel_coord_transform(cell_addr, w, h, dst_crs=crs, native=native)

    w = maxW if w == 'max' else w
    h = maxH if h == 'max' else h

    bb = 0.5 - 1e-8
    x = np.linspace(-bb, w - 1 + bb, w+1)
    y = np.linspace(-bb, h - 1 + bb, h+1)
    x, y = polygon_path(x, y)

    u, v = tr(x, y)

    ax = _get_ax(ax)
    ax.plot(u, v, style, **kwargs)


def plot_roi(roi, style='-', ax=None, south_square=0, north_square=0, **kwargs):

    f = dg.mk_display_helper(south_square=south_square,
                             north_square=north_square)

    _, extents = f(roi.addr, roi.shape)
    return plot_bbox(extents, style=style, ax=ax, **kwargs)


def index_to_rgb(im, palette, alpha=None):
    alpha_ch = () if alpha is None else (0xFF,)

    def to_rgb(v):
        return tuple((v >> (i*8)) & 0xFF for i in [2, 1, 0]) + alpha_ch

    palette = np.vstack(map(to_rgb, palette)).astype('uint8')

    im_c = palette[im]
    if alpha is not None:
        im_c[im == alpha, 3] = 0

    return im_c

import numpy as np
import matplotlib.pyplot as plt

from .tools import polygon_path
from .dggs import DGGS

dg = DGGS()


def _get_ax(ax):
    return ax if ax else plt.gca()


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

import numpy as np
import xarray as xr
import matplotlib.figure
import matplotlib.pyplot as plt

from .tools import polygon_path
from . import DGGS

dg = DGGS()


class ReprWrapper(object):
    """ Turns string into object such that `repr(ReprWrapper(s)) == s`
    """
    def __init__(self, val):
        self._val = val

    def __repr__(self):
        return self._val

    def __str__(self):
        return self._val


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
    tr, *_ = dg.pixel_coord_transform(addr, native=True, no_offset=True)
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


def _to_roi(x):
    if isinstance(x, (str, DGGS.Address)):
        return DGGS.ROI(x)
    return x


def _to_addr(x):
    if hasattr(x, 'addr'):
        return x.addr
    if isinstance(x, str):
        return DGGS.Address(x)
    return x


class DgDraw(object):
    def __init__(self, ax,
                 north_square=0,
                 south_square=0,
                 dg=DGGS()):
        if ax == 'new':
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1])
        if isinstance(ax, matplotlib.figure.Figure):
            ax = ax.add_axes([0, 0, 1, 1])

        self._ax = ax
        self._dg = dg
        self._helper = dg.mk_display_helper(south_square=south_square,
                                            north_square=north_square)

    def imshow(self, data, band_idx=0, reset_axis=True, **kwargs):
        if not isinstance(data, list):
            data = [data]

        ax = self._ax
        extents = None if reset_axis else ax.get_xlim() + ax.get_ylim()

        for ds in data:
            addr = ds.addr

            if isinstance(ds, xr.Dataset):
                if is_rgba(ds):
                    im = as_rgba(ds)
                elif is_rgb(ds):
                    im = as_rgb(ds)
                else:
                    im = list(ds.data_vars.values())[band_idx].values
            elif isinstance(ds, xr.DataArray):
                im = ds.values
            elif isinstance(ds, DGGS.Image):
                im = ds.value
            else:
                raise ValueError('Expect one of: xarray.Data{set,Array}, DGGS.Image')

            im, ee = self._helper(addr, im)
            extents = merge_extents(extents, ee)
            ax.imshow(im, extent=ee, **kwargs)

        ax.set_xlim(*extents[:2])
        ax.set_ylim(*extents[2:])
        return self

    def roi(self, roi, style='-', **kwargs):
        roi = _to_roi(roi)

        _, extents = self._helper(roi.addr, roi.shape)
        plot_bbox(extents, style=style, ax=self._ax, **kwargs)
        return self

    def qshow(self, q, max_level=None, grid_style=None, **kwargs):
        from . import mask_from_addresses
        im = mask_to_float(mask_from_addresses(q))

        params = dict(cmap='spring',
                      reset_axis=False,
                      alpha=0.1)

        params.update(kwargs)
        self.imshow(im, **params)

        if grid_style is None:
            grid_style = dict(style='w-', alpha=0.7)

        def should_plot(a):
            if max_level is None:
                return True
            return len(a)-1 <= max_level

        for addr in q:
            if should_plot(addr):
                self.roi(addr, **grid_style)

        return self

    def annotate(self, txt, addr, **kwargs):
        addr = _to_addr(addr)
        cx, cy = cell_center(addr)
        self._ax.annotate(txt, xy=(cx, cy), **kwargs)
        return self

    def scatter(self, addrs, **kwargs):
        cx, cy = np.r_[[cell_center(a) for a in addrs]].T
        self._ax.scatter(cx, cy, **kwargs)
        return self

    def hide_axis(self):
        hide_axis(self._ax)
        return self

    def zoom(self, roi):
        roi = _to_roi(roi)
        _, extents = self._helper(roi.addr, roi.shape)
        self._ax.axis(extents)
        return self

    @property
    def raw(self):
        return self._ax

    @property
    def figure(self):
        return self._ax.figure


def save_axis_tight(fname, ax=None, **kwargs):
    ax = _get_ax(ax)
    fig = ax.figure

    fig.savefig(fname,
                bbox_inches=ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()),
                pad_inches=0,
                transparent=True,
                **kwargs)


def mask_to_float(mm):
    """ True -> 1.0
        False -> nan

    Useful for visualising masks
    """
    xx = mm.value.astype('float32')
    xx[xx == 0] = np.nan
    return type(mm)(xx, mm.addr)


def addr_mask_repr(addr, n=4):
    n_cells = len(addr)
    n = min(n, n_cells)

    ss = '{}:\n   {} ...\n   {}'.format(n_cells,
                                        ','.join(addr[:n]),
                                        ','.join(addr[-n:]))
    return ReprWrapper(ss)

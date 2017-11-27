import numpy as np
import xarray as xr
import h5py
from collections import namedtuple

from . import DGGS

BandInfo = namedtuple('BandInfo', ['dtype', 'nodata', 'block', 'name'])
GeoFileInfo = namedtuple('GeoFileInfo', ['crs', 'affine', 'shape', 'bands'])


def _h5_parse_structure(f):
    sites = {}
    bands = []

    def valid_address(addr: str):
        code, *digits = addr
        return (code in 'SNOPQR' and
                set(digits).issubset('012345678'))  # TODO: assumes 3x3 dggs

    def add(addr, shape):
        v = sites.get(addr)
        if v and v != shape:
            raise RuntimeError('Incompatible shapes detected')
        else:
            sites[addr] = shape

    for band, g in f.items():
        if not isinstance(g, h5py.Group):
            continue

        count = 0

        for addr, ds in g.items():
            if valid_address(addr) and isinstance(ds, h5py.Dataset):
                add(addr, ds.shape)
                count += 1

        if count > 0:
            bands.append(band)

    return bands, sites


def h5_load(fname, bands=None, dg=DGGS()):

    def read_bands(f, addr, shape, bands):
        h, w = shape[:2]
        roi = dg.ROI(addr, w, h)
        coords = dg.xy_from_roi(roi)[::-1]

        dims = ('y', 'x')

        def read(band):
            path = band + '/' + addr
            ds = f.get(path)
            if ds is None:
                # TODO: need to know dtype and nodata value for the band
                raise NotImplementedError("Currently only support homogeneous data across bands")
            else:
                if ds.shape != shape:
                    raise NotImplementedError("Currently only support homogeneous data across bands")

                dd = np.empty(shape, dtype=ds.dtype)
                ds.read_direct(dd)

                # TODO: nodata
                return xr.DataArray(dd,
                                    dims=dims,
                                    name=band,
                                    coords=coords,
                                    attrs=dict(addr=addr))

        return xr.Dataset({band: read(band) for band in bands},
                          attrs=dict(addr=addr))

    with h5py.File(fname, 'r') as f:
        bands_, sites = _h5_parse_structure(f)
        if bands is None:
            bands = bands_
        else:
            # TODO: verify that requested bands are present in a file
            pass

        return [read_bands(f, addr, shape, bands)
                for (addr, shape) in sites.items()]


class H5Writer(object):
    def __init__(self, fname, chunk_size=3**5):
        self.fname = fname
        self._chunk_size = chunk_size
        self._f = None
        self._opts = dict(compression='gzip',
                          shuffle=True)

    def _chunks(self, shape):
        a = min(shape[0], self._chunk_size)
        b = min(shape[1], self._chunk_size)
        return (a, b) + shape[2:]

    def __enter__(self):
        self._f = h5py.File(self.fname, 'w')
        return self

    def __exit__(self, t, v, traceback):
        self._f.close()
        self._f = None

    def __call__(self, addr, band, data, nodata=None):
        f = self._f
        g = f.get(band)

        if g is None:
            g = f.create_group(band)
        elif not isinstance(g, h5py.Group):
            raise IOError('TODO: fix error message')

        if not isinstance(addr, str):
            addr = str(addr)

        ds = g.create_dataset(addr,
                              data=data,
                              chunks=self._chunks(data.shape),
                              fillvalue=nodata,
                              **self._opts)
        if ds.ndim >= 2:
            ds.dims[0].label = 'y'
            ds.dims[1].label = 'x'


def h5_save(fname, datasets, chunk_size=3**5):
    if not isinstance(datasets, (tuple, list)):
        datasets = [datasets]

    with H5Writer(fname, chunk_size=chunk_size) as write:
        for ds in datasets:
            assert hasattr(ds, 'addr')

            for name, da in ds.data_vars.items():
                nodata = da.attrs.get('nodata', None)
                write(ds.addr, name, da.values, nodata=nodata)

    return True


def slurp(fname, proc=None, keep_eol=False):
    import gzip
    import lzma

    _open = open

    if fname.endswith('.gz'):
        _open = gzip.open
    if fname.endswith('.xz'):
        _open = lzma.open

    def mk_proc(proc):
        maybe_strip = (lambda s: s.rstrip('\n')) if keep_eol is False else (lambda s: s)

        if proc is None:
            return maybe_strip
        else:
            return lambda s: proc(maybe_strip(s))

    with _open(fname, 'rt') as f:
        return list(map(mk_proc(proc), f.readlines()))


def dump_text(txt, fname=None):
    import gzip
    import lzma
    import sys

    eol = '\n'

    def write_to(txt, f):
        if isinstance(txt, str):
            f.write(txt)
        else:
            f.writelines(map(lambda x: str(x) + eol, txt))

    if fname is None:
        write_to(txt, sys.stdout)
        return True

    _open = open

    if fname.endswith('.gz'):
        _open = gzip.open
    if fname.endswith('.xz'):
        _open = lzma.open

    with _open(fname, 'wt') as f:
        write_to(txt, f)

    return True


def load_shapes(fname, pred=lambda _: True, with_attributes=True):
    import fiona
    from shapely.geometry import shape

    def mk_shape(g):
        sh = shape(g['geometry'])
        if with_attributes:
            sh.attrs = g['properties'].copy()
        return sh

    with fiona.open(fname, 'r') as f:
        shapes = [mk_shape(g)
                  for g in f.values() if pred(g)]

        return shapes, f.crs


def load_polygons(fname):
    return load_shapes(fname, lambda g: g['geometry']['type'] == 'Polygon')


def save_png(fname, im, bgr=False, binary=None):
    import cv2

    if im.ndim == 3 and bgr is False:
        _, _, nc = im.shape
        if nc == 3:
            im = im[:, :, ::-1]  # Convert to BGR
        elif nc == 4:
            im = im[:, :, [2, 1, 0, 3]]  # Convert to BGRA

    png_opts = (cv2.IMWRITE_PNG_COMPRESSION, 9)

    if im.dtype == np.bool:
        im = im.astype('uint8')
        binary = True if binary is None else binary

    if binary:
        png_opts = png_opts + (cv2.IMWRITE_PNG_BILEVEL, 1)

    return cv2.imwrite(fname, im, png_opts)


def geo_file_info(fname, band_names=None):
    import rasterio

    def band_name(idx):
        if band_names is None:
            return None
        return band_names[idx]

    def info(f):
        def band_info(idx):
            return BandInfo(f.dtypes[idx], f.nodatavals[idx], f.block_shapes[idx], band_name(idx))

        bands = [band_info(i) for i in range(f.count)]
        return GeoFileInfo(f.crs.to_dict(), f.affine, f.shape, bands)

    if isinstance(fname, str):
        with rasterio.open(fname, 'r') as f:
            return info(f)
    return info(fname)


def geo_load(fname, fix_nodata=True, band_names=None):
    import rasterio

    def bad_nodata(band):
        if np.dtype(band.dtype).kind == 'f':
            if (band.nodata is not None) and (not np.isnan(band.nodata)):
                return True
        return False

    def fix_band_info(band):
        T = type(band)
        band = band._asdict()
        band['nodata'] = np.nan
        return T(**band)

    with rasterio.open(fname, 'r') as f:
        info = geo_file_info(f, band_names=band_names)
        bands = []

        for i, band in enumerate(info.bands):
            data = f.read(i+1)

            if fix_nodata and bad_nodata(band):
                data[data == band.nodata] = np.nan
                info.bands[i] = fix_band_info(band)

            bands.append(data)

        return info, bands

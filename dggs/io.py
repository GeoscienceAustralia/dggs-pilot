import numpy as np
import xarray as xr
import h5py


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


def h5_load(fname, bands=None):

    def read_bands(f, addr, shape, bands):
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
                return dd

        dims = ('y', 'x')
        return xr.Dataset({band: (dims, read(band)) for band in bands})

    with h5py.File(fname, 'r') as f:
        bands_, sites = _h5_parse_structure(f)
        if bands is None:
            bands = bands_
        else:
            # TODO: verify that requested bands are present in a file
            pass

        return {addr: read_bands(f, addr, shape, bands)
                for (addr, shape) in sites.items()}

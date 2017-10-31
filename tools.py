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


def apply_affine(A, x, y):
    assert x.shape == y.shape
    assert hasattr(A, '__mul__')

    sh = x.shape

    x, y = A*np.vstack([x.ravel(), y.ravel()])
    return x.reshape(sh), y.reshape(sh)


def spread_num_samples(v):
    v_int = np.floor(v).astype('uint32')
    n_partials = int(v.sum() - v_int.sum())

    a = np.fmod(v, 1)
    a /= a.sum()

    ii = np.random.choice(v.shape[0], n_partials, p=a, replace=False)
    v_int[ii] += 1

    return v_int


def gen_pts_from_distribution(im, affine, noshuffle=False):
    iy, ix = np.where(im > 0)

    vv = spread_num_samples(im[iy, ix])

    n_total = vv.sum()
    off = 0
    xy = np.zeros((2, n_total), dtype='float32')

    for v, x, y in zip(vv, ix, iy):
        if v == 0:
            continue

        pts = np.random.uniform(0, 1, size=(2, v)) + np.c_[[x, y]]

        xy[:, off:off+v] = affine*pts

        off += v

    if noshuffle:
        return xy

    return xy[:, np.random.choice(n_total, n_total, replace=False)]


def convert_global(src_file, dst_file, scale_level, band_names=None, combine_bands=False):
    import rasterio
    import h5py
    from .dggs import DGGS

    if isinstance(band_names, str):
        band_names = [band_names]

    def get_band_name(i):
        if band_names is None:
            return 'band{}'.format(i+1)

        return band_names[i]

    affine, src_crs = None, None
    bands = []

    with rasterio.open(src_file, 'r') as f:
        affine = f.affine
        bands = [f.read(i+1) for i in range(f.count)]
        if combine_bands:
            bands = [np.dstack(bands)]
        src_crs = f.crs

    num_bands = len(bands)

    print('CRS:', src_crs)

    dg = DGGS()

    cells = {}

    for code in 'SNOPQR':
        addr = code + '0'*scale_level
        print('Processing:' + addr)

        wrp = dg.mk_warper(addr)

        cells[addr] = [wrp(b, affine) for b in bands]

    chunks = (3**5, 3**5)  # TODO: probably needs to be dynamic

    opts = dict(compression='gzip',
                shuffle=True)

    with h5py.File(dst_file, 'w') as f:
        for i in range(num_bands):
            band_name = get_band_name(i)
            print('Saving band: {band_name}'.format(band_name=band_name))

            g = f.create_group(band_name)
            for (addr, bands) in cells.items():
                band = bands[i]
                chunks_ = chunks + band.shape[2:]
                g.create_dataset(addr, data=band, chunks=chunks_, **opts)

        f.close()

    return cells


def convert(src_file, dst_file, scale_level, band_names=None, inter=None):
    import rasterio
    import h5py
    from .dggs import DGGS

    if isinstance(band_names, str):
        band_names = [band_names]

    def get_band_name(i):
        if band_names is None:
            return 'band{}'.format(i+1)

        return band_names[i]

    def read_band(f, i):
        band = f.read(i+1)
        nodata = f.nodatavals[i]

        if band.dtype.kind == 'f' and np.isfinite(nodata):
            print('Fixing crazy nodata value to be NaN')
            band[band == nodata] = np.nan
            nodata = np.nan

        return band, nodata

    def get_chunks(shape, xy_size):
        a = min(shape[0], xy_size)
        b = min(shape[1], xy_size)
        return (a, b) + shape[2:]

    def has_valid_data(bands):
        for band, nodata in zip(bands, nodatavals):
            if np.isnan(nodata):
                empty = np.isnan(band).all()
            else:
                empty = (band == nodata).all()

            if not empty:
                return True

        return False

    affine, src_crs, src_x, src_y = None, None, None, None
    bands = []

    with rasterio.open(src_file, 'r') as f:
        affine = f.affine
        nodatavals = []
        bands = []

        for i in range(f.count):
            band, nodata = read_band(f, i)
            bands.append(band)
            nodatavals.append(nodata)

        src_crs = f.crs
        src_x = np.linspace(f.bounds.left, f.bounds.right, f.width)
        src_y = np.linspace(f.bounds.top, f.bounds.bottom, f.height)

    num_bands = len(bands)
    dg = DGGS()

    cells = {}
    boundary_x, boundary_y = polygon_path(src_x, src_y)

    for addr, w, h in dg.compute_overlap(scale_level, boundary_x, boundary_y, src_crs):
        print('Processing:' + addr, w, h)
        wrp = dg.mk_warper(addr, w, h, src_crs=src_crs)
        out = []

        for band, nodata in zip(bands, nodatavals):
            im = wrp(band, affine, nodata, inter=inter)
            out.append(im)

        if has_valid_data(out):
            cells[addr] = out

    chunk_size = 3**5  # TODO: probably needs to be dynamic

    opts = dict(compression='gzip',
                shuffle=True)

    with h5py.File(dst_file, 'w') as f:
        for i in range(num_bands):
            band_name = get_band_name(i)
            print('Saving band: {band_name}'.format(band_name=band_name))

            g = f.create_group(band_name)
            for (addr, bands) in cells.items():
                band = bands[i]
                nodata = nodatavals[i]
                chunks = get_chunks(band.shape, chunk_size)
                g.create_dataset(addr, data=band,
                                 chunks=chunks,
                                 fillvalue=nodata,
                                 **opts)

        f.close()

    return cells

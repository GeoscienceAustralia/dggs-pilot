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

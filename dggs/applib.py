import numpy as np
from . import DGGS


def convert_file(src_file, dst_file, scale_level, band_names=None, inter=None, align_by=5):
    from .io import geo_load, h5_save
    from .warp import warp_all

    dg = DGGS()

    info, bands = geo_load(src_file, band_names=band_names, fix_nodata=True)

    dst_roi = dg.roi_from_geo(info, scale_level, align_by=align_by)

    return h5_save(dst_file,
                   warp_all(info, bands,
                            dst_roi=dst_roi,
                            inter=inter))


def convert_file_global(src_file, dst_file, scale_level, band_names=None, inter=None, align_by=5):
    from .io import geo_load, h5_save
    from .warp import warp_all

    info, bands = geo_load(src_file, band_names=band_names, fix_nodata=True)

    dst_roi = []
    s = 3**scale_level
    for c in 'NSOPQR':
        roi = DGGS.ROI(c + '0'*scale_level, s, s)
        dst_roi.append(roi)

    return h5_save(dst_file,
                   warp_all(info, bands,
                            dst_roi=dst_roi,
                            inter=inter))


def abs_data_convert(src_file, dst_file, scale_level=9, align_by=5):
    from .io import geo_load, h5_save
    from .warp import warp_all

    dg = DGGS()

    # Load source file
    band_names = ['population_count']
    info, bands = geo_load(src_file, band_names=band_names, fix_nodata=True)

    # Warp to a particular scale
    dst_roi = dg.roi_from_geo(info, scale_level, align_by=align_by)
    regions = warp_all(info, bands,
                       dst_roi=dst_roi,
                       inter='area')

    # Normalise such that output sum is roughly the same as input sum
    ref_sum = np.nansum(bands[0])
    actual_sum = sum([b.population_count.sum().values.item() for b in regions])
    correction_scale = ref_sum/actual_sum

    for ds in regions:
        ds *= correction_scale

    # serialise to disk
    return h5_save(dst_file, regions)


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


def pd_naive_overlap_str(query_ddresses, df):
    mm = np.zeros(len(df), dtype=np.bool)

    for a in query_ddresses:
        mm |= np.r_[[addr.startswith(a) for addr in df.addr.values]]

    return mm


def pd_naive_overlap(query, df):
    def a64_range(addr):
        if isinstance(addr, str):
            addr = DGGS.Address(addr)

        assert addr.scale <= 15

        pad = (15-addr.scale)*4

        a64 = addr.a64
        a_min = a64 - ((1 << pad) - 1)
        a_max = a64 + 1
        return (a_min, a_max)

    mm = np.zeros(len(df), dtype=np.bool)

    for a in query:
        amin, amax = a64_range(a)
        mm |= ((df.addr64 >= amin) & (df.addr64 < amax)).values

    return mm

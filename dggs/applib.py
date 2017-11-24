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

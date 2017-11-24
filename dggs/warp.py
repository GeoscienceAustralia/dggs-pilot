import numpy as np
import xarray as xr
from . import DGGS

dg = DGGS()


def warp_all(info, bands,
             dst_roi=None, scale=None,
             inter=None,
             band_names=None,
             prune_empty=True):

    if dst_roi is None:
        assert scale is not None
        dst_roi = dg.roi_from_geo(info, scale)

    if isinstance(dst_roi, DGGS.ROI):
        dst_roi = [dst_roi]

    dims = ('y', 'x')

    def warp_roi(dst_roi):
        wrp = dg.mk_warper(dst_roi, src_crs=info.crs)
        coords = dg.xy_from_roi(dst_roi)[::-1]

        for idx, band in enumerate(bands):
            band_name = info.bands[idx].name
            nodata = info.bands[idx].nodata

            if band_name is None:
                band_name = 'band_{idx}'.format(idx=idx)

            im = wrp(band, info.affine, nodata, inter=inter)

            attrs = {}
            if nodata is not None:
                attrs['nodata'] = nodata

            yield xr.DataArray(im, dims=dims, coords=coords, name=band_name, attrs=attrs)

    def da_is_empty(da):
        nodata = da.attrs.get('nodata', None)
        if nodata is None:
            return False
        if np.isnan(nodata):
            return da.isnull().all().values.item()
        else:
            return (da == nodata).all().values().item()

    def ds_is_empty(ds):
        for da in ds.data_vars.values():
            if not da_is_empty(da):
                return False
        return True

    oo = []

    for roi in dst_roi:
        ds = xr.Dataset({da.name: da for da in warp_roi(roi)},
                        attrs=dict(addr=str(roi.addr)))

        if prune_empty and ds_is_empty(ds):
            # skipping this dataset -- no valid data present and were asked to prune
            pass
        else:
            oo.append(ds)

    return oo

import pathlib

_data_dir = pathlib.Path('./data')


def get_path(resource):
    return str(_data_dir/resource)


def act_suburbs():
    from .io import load_polygons

    suburbs, crs = load_polygons(get_path('ACT_LOCALITY_POLYGON_shp/ACT_LOCALITY_POLYGON_shp.shp'))

    for s in suburbs:
        s.suburb = s.attrs['ACT_LOCA_2']

    suburbs = {s.suburb: s for s in suburbs}
    return suburbs, crs


def act_nexis(keys=None):
    from .io import load_shapes
    import pandas as pd

    file = 'NEXIS/NEXIS_V9_Residential_ACT.shp'

    if keys is None:
        keys = 'VALUE PEOPLE ADDRESSES CONTENTS FOOTPRINT FLOOR_AREA OWN'.split()

    def to_dict(p):
        oo = dict(x=p.x, y=p.y)
        oo.update(**{k.lower(): p.attrs.get(k) for k in keys})
        return oo

    pts, crs = load_shapes(get_path(file))

    columns = 'x y'.split() + [k.lower() for k in keys]
    data = pd.DataFrame([to_dict(p) for p in pts], columns=columns)
    return data, crs


_data_sets = {
    'act-suburbs': act_suburbs,
    'act-nexis': act_nexis
}


def get_by_name(name=None, **kwargs):
    if name is None:
        return list(_data_sets.keys())

    f = _data_sets.get(name)

    if f is None:
        return None

    return f(**kwargs)


def act_suburb_mask(scale_level, mode='text'):
    """ Returns a function that maps ACT suburb name (all capitals) to a DGGS mask.
    """
    from . import shape_to_mask, mask_to_addresses
    suburbs, crs = get_by_name('act-suburbs')

    def get(name=None, scale_level=scale_level, mode=mode):
        if name is None:
            return list(suburbs.keys())

        poly = suburbs.get(name)
        if poly is None:
            return None

        im = shape_to_mask(suburbs[name], crs=crs, scale_level=scale_level)
        if mode == 'text':
            return mask_to_addresses(im)

        return im
    return get

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

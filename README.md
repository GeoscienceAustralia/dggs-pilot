# Introduction

DGGS pilot. Currently implements only rHealPix 3x3 DGGS.

Features:

- Raster storage using hdf5
- Conversion from GeoTiff (or any format GDAL supports)
- Shapefile polygons -> list of addresses

# Installation

- This was only tested with conda
- Needs python 3.6+ (uses format strings in some places)

On Mac or Linux:

```
    conda env create -f env.yaml
    source activate dggs
    pip install --no-deps -e .
    ./launch-jupyter.sh
```

On Windows

```
    conda env create -f env.yaml
    activate dggs
    pip install --no-deps -e .
    launch-jupyter.bat
```


# Getting sample data

Sample data is not included in the repository, some of it might be sensitive, other might have re-distribution consttraints. Data is distributed separately on request.

Extract password protected `dggs-sample-data.zip` using provided password. Move files from `data/*` to `data/` folder in this repoistory.


# Running notebooks

First run these notebooks

- `io-convert-cbr`
- `io-convert-abs`

This will convert some test data to DGGS. You can review by running these notebooks

- `io-h5-read-cbr`
- `io-h5-read-abs`

Then run

- `nexis-dataset`

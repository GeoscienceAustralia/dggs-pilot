from setuptools import setup, find_packages

setup(
    name='dggs',
    version='0.1',
    license='Apache License 2.0',  # TODO: copied from datacube
    package=find_packages(),
    author='Geoscience Australia',
    description='DGGS prototype/investigation',
    install_requires=['xarray',
                      'numpy',
                      'h5py',
                      'pandas',
                      'rasterio',
                      'opencv-python',
                      'fiona',
                      'shapely',
                      'pydash',
                      ],
    tests_require=['pytest'],
)

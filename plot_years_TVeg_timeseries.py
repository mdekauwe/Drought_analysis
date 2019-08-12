#!/usr/bin/env python
"""
Plot Jan to Dec TVeg
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (25.07.2019)"
__email__ = "mdekauwe@gmail.com"

import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import cartopy.crs as ccrs
import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import sys
import matplotlib.ticker as mticker
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import AxesGrid

def main(fname, plot_dir):

    ds = xr.open_dataset(fname)

    TVeg = ds.TVeg[:,101:125,639:668]
    LAI = ds.LAI[:,101:125,639:668]
    plt.scatter(22, 20, s=5, c='blue', marker='o')  # good
    plt.scatter(23, 20, s=5, c='red', marker='o')   # bad
    plt.imshow(TVeg[0,:,:] * 86400)

    plt.colorbar()
    plt.show()

    #(12, 24, 29)
    #print(TVeg.shape)
    good = TVeg[:,20,22]
    bad = TVeg[:,20,23]
    good = LAI[:,20,22]
    bad = LAI[:,20,23]


    print(bad)
    #plt.plot(good * 86400, "-b", label="good")
    #plt.plot(bad * 86400, "-r", label="bad")
    plt.plot(good, "-b", label="good")
    plt.plot(bad, "-r", label="bad")
    plt.legend()
    plt.show()



if __name__ == "__main__":

    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    fname = "outputs/cable_out_1995.nc"

    main(fname, plot_dir)

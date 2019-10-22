#!/usr/bin/env python
"""
Plot DJF for each year of the Millennium drought
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

    row = 292
    col = 590

    ds = xr.open_dataset(fname)
    lat = ds.y.values
    lon = ds.x.values
    bottom, top = lat[0], lat[-1]
    left, right = lon[0], lon[-1]

    plc = ds.plc[:,0,:,:].values
    print(plc.shape)

    #plt.imshow(plc[0,:,:])
    #plt.colorbar()
    #plt.show()

    plt.plot(plc[:,row,col])
    plt.show()

if __name__ == "__main__":

    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    #fname = "outputs/min_plc.nc"
    fname = "outputs/all_yrs_plc.nc"
    main(fname, plot_dir)

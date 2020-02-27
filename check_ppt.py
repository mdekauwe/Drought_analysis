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
from calendar import monthrange

def main(fname, plot_dir):

    ds = xr.open_dataset(fname)
    lat = ds.y.values
    lon = ds.x.values
    bottom, top = lat[0], lat[-1]
    left, right = lon[0], lon[-1]

    yr_one = ds.Rainf[0:12,:,:]
    year = 2000
    sec_2_day = 86400.0
    for month in np.arange(1, 13):

        days_in_month = monthrange(year, month)[1]
        conv = sec_2_day * days_in_month
        yr_one[month-1,:,:] *= conv

    annual_ppt = yr_one.sum(axis=0)
    annual_ppt = np.where(annual_ppt == 0.0, np.nan, annual_ppt)
    plt.imshow(annual_ppt[100:350,550:841])
    plt.colorbar()
    plt.show()


if __name__ == "__main__":

    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    fname = "outputs/all_yrs_CMI.nc"

    main(fname, plot_dir)

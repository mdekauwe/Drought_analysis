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

def main(fname1, fname2, plot_dir):

    ds = xr.open_dataset(fname1)

    nmonths, nrows, ncols = ds.Rainf.shape

    cmi = np.zeros((nmonths, nrows, ncols))
    cnt = 0
    sec_2_day = 86400.0
    for year in np.arange(2000, 2010):
        for month in np.arange(1, 13):
            days_in_month = monthrange(year, month)[1]
            conv = sec_2_day * days_in_month
            cmi[cnt,:,:] = (ds.Rainf[cnt,:,:] * conv) - \
                            (ds.Evap[cnt,:,:] * conv)


            cnt = cnt + 1

    cmi = np.sum(cmi, axis=0)

    # just keep deficit areas
    cmi = np.where(cmi >= 300., np.nan, cmi)

    ds = xr.open_dataset(fname2)

    plc = ds.plc[:,0,:].values
    plc = np.nanmean(plc, axis=0)
    plc = np.where(plc >= 88., 88, plc)


    fig = plt.figure(figsize=(6, 6))
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['font.size'] = 16
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16

    ax = fig.add_subplot(111)
    ax.plot(cmi/9., plc, "ko")

    ax.set_ylabel('Min PLC (%)')
    ax.set_xlabel('PPT-ET (mm yr$^{-1}$)') # mm 9yrs-1



    ofname = os.path.join(plot_dir, "cmi_plc_correlation.png")
    fig.savefig(ofname, dpi=150, bbox_inches='tight',
                pad_inches=0.1)



if __name__ == "__main__":

    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    fname1 = "outputs/all_yrs_CMI.nc"
    fname2 = "outputs/all_yrs_plc.nc"

    main(fname1, fname2, plot_dir)

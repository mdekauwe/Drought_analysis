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
import pandas as pd

def main(plot_dir):

    # layer thickness
    zse = np.array([.022, .058, .154, .409, 1.085, 2.872])


    plc_rf_all = np.zeros(0)
    plc_wsf_all = np.zeros(0)
    plc_dsf_all = np.zeros(0)
    plc_grw_all = np.zeros(0)
    plc_saw_all =  np.zeros(0)

    sw_rf_all = np.zeros(0)
    sw_wsf_all = np.zeros(0)
    sw_dsf_all = np.zeros(0)
    sw_grw_all = np.zeros(0)
    sw_saw_all = np.zeros(0)

    start_yr = 2000
    end_yr = 2010
    nyears = (end_yr - start_yr) + 1
    nmonths = 12

    fdir = "outputs"
    fname = os.path.join(fdir, "cable_out_2000.nc")
    ds = xr.open_dataset(fname)
    iveg = ds["iveg"][:,:].values
    idx_rf = np.argwhere(iveg == 18.0)
    idx_wsf = np.argwhere(iveg == 19.0)
    idx_dsf = np.argwhere(iveg == 20.0)
    idx_grw = np.argwhere(iveg == 21.0)
    idx_saw = np.argwhere(iveg == 22.0)

    plc_rf_all = np.zeros((nyears * nmonths, len(idx_rf)))
    plc_wsf_all = np.zeros((nyears * nmonths, len(idx_wsf)))
    plc_dsf_all = np.zeros((nyears * nmonths, len(idx_dsf)))
    plc_grw_all = np.zeros((nyears * nmonths, len(idx_grw)))
    plc_saw_all =  np.zeros((nyears * nmonths, len(idx_saw)))

    sw_rf_all = np.zeros((nyears * nmonths, len(idx_rf)))
    sw_wsf_all = np.zeros((nyears * nmonths, len(idx_wsf)))
    sw_dsf_all = np.zeros((nyears * nmonths, len(idx_dsf)))
    sw_grw_all = np.zeros((nyears * nmonths, len(idx_grw)))
    sw_saw_all =  np.zeros((nyears * nmonths, len(idx_saw)))

    nyear = 0
    cnt = 0
    for year in np.arange(start_yr, end_yr):
        print(year)
        fdir = "outputs"
        fname = os.path.join(fdir, "cable_out_%d.nc" % (year))
        ds = xr.open_dataset(fname)
        plc_vals = ds["plc"][:,0,:,:].values

        """
        SoilMoist1 = ds["SoilMoist"][:,0,:,:].values * zse[0]
        SoilMoist2 = ds["SoilMoist"][:,1,:,:].values * zse[1]
        SoilMoist3 = ds["SoilMoist"][:,2,:,:].values * zse[2]
        SoilMoist4 = ds["SoilMoist"][:,3,:,:].values * zse[3]
        SoilMoist5 = ds["SoilMoist"][:,4,:,:].values * zse[4]
        SoilMoist6 = ds["SoilMoist"][:,5,:,:].values * zse[5]
        sw = (SoilMoist1 + SoilMoist2 + SoilMoist3 + \
                SoilMoist4 + SoilMoist5 + SoilMoist6 ) / np.sum(zse)
        """
        SoilMoist1 = ds["SoilMoist"][:,0,:,:].values * zse[0]
        SoilMoist2 = ds["SoilMoist"][:,1,:,:].values * zse[1]
        SoilMoist3 = ds["SoilMoist"][:,2,:,:].values * zse[2]
        SoilMoist4 = ds["SoilMoist"][:,3,:,:].values * zse[3]

        sw = (SoilMoist1 + SoilMoist2 + \
              SoilMoist3 + SoilMoist4) / np.sum(zse[0:4])

        idx = nyear + cnt

        plc_rf = np.zeros((12,len(idx_rf)))
        sw_rf = np.zeros((12,len(idx_rf)))
        for i in range(len(idx_rf)):
            (row, col) = idx_rf[i]
            plc_rf[:,i] = plc_vals[:,row,col]
            sw_rf[:,i] = sw[:,row,col]
        plc_rf_all[idx:(idx+12),:] = plc_rf
        sw_rf_all[idx:(idx+12),:] = sw_rf

        plc_wsf = np.zeros((12,len(idx_wsf)))
        sw_wsf = np.zeros((12,len(idx_wsf)))
        for i in range(len(idx_wsf)):
            (row, col) = idx_wsf[i]
            plc_wsf[:,i] = plc_vals[:,row,col]
            sw_wsf[:,i] = sw[:,row,col]
        plc_wsf_all[idx:(idx+12),:] = plc_wsf
        sw_wsf_all[idx:(idx+12),:] = sw_wsf

        plc_dsf = np.zeros((12,len(idx_dsf)))
        sw_dsf = np.zeros((12,len(idx_dsf)))
        for i in range(len(idx_dsf)):
            (row, col) = idx_dsf[i]
            plc_dsf[:,i] = plc_vals[:,row,col]
            sw_dsf[:,i] = sw[:,row,col]
        plc_dsf_all[idx:(idx+12),:] = plc_dsf
        sw_dsf_all[idx:(idx+12),:] = sw_dsf

        plc_grw = np.zeros((12,len(idx_grw)))
        sw_grw = np.zeros((12,len(idx_grw)))
        for i in range(len(idx_grw)):
            (row, col) = idx_grw[i]
            plc_grw[:,i] = plc_vals[:,row,col]
            sw_grw[:,i] = sw[:,row,col]
        plc_grw_all[idx:(idx+12),:] = plc_grw
        sw_grw_all[idx:(idx+12),:] = sw_grw

        plc_saw = np.zeros((12,len(idx_saw)))
        sw_saw = np.zeros((12,len(idx_saw)))
        for i in range(len(idx_saw)):
            (row, col) = idx_saw[i]
            plc_saw[:,i] = plc_vals[:,row,col]
            sw_saw[:,i] = sw[:,row,col]
        plc_saw_all[idx:(idx+12),:] = plc_saw
        sw_saw_all[idx:(idx+12),:] = sw_saw


        nyear += 1
        cnt += 12

    #from matplotlib.pyplot import cm
    #colours = cm.Set2(np.linspace(0, 1, 5))
    #colours = cm.get_cmap('Set2')
    months = []
    dates = []
    years = []
    periods = (end_yr - start_yr + 1) * 12
    date = pd.date_range('01/01/%d' % (start_yr), periods=periods, freq ='M')

    for i in range(len(idx_saw)):
        mth_cnt = 1
        for j in range(nyears * nmonths):
            if plc_saw_all[j,i] >= 80:
                months.append(mth_cnt)
                dates.append(date[j])
                years.append(2000 + ((mth_cnt-1) / 12))

            mth_cnt += 1


    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111)
    #ax.hist(months)
    ax.hist(years, bins=20)
    ax.xaxis.set_ticks(np.arange(start_yr, end_yr+1, 1))
    ax.set_xlim(2000, 2011)
    odir = "plots"
    plt.savefig(os.path.join(odir, "saw_hist_plc_over_80_when.pdf"),
                bbox_inches='tight', pad_inches=0.1)


if __name__ == "__main__":

    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    main(plot_dir)

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

def main(plot_dir):

    plc_rf_all = np.zeros(0)
    plc_wsf_all = np.zeros(0)
    plc_dsf_all = np.zeros(0)
    plc_grw_all = np.zeros(0)
    plc_saw_all =  np.zeros(0)

    start_yr = 2000
    end_yr = 2004
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

    et_rf_all = np.zeros((nyears * nmonths, len(idx_rf)))
    et_wsf_all = np.zeros((nyears * nmonths, len(idx_wsf)))
    et_dsf_all = np.zeros((nyears * nmonths, len(idx_dsf)))
    et_grw_all = np.zeros((nyears * nmonths, len(idx_grw)))
    et_saw_all =  np.zeros((nyears * nmonths, len(idx_saw)))

    ppt_rf_all = np.zeros((nyears * nmonths, len(idx_rf)))
    ppt_wsf_all = np.zeros((nyears * nmonths, len(idx_wsf)))
    ppt_dsf_all = np.zeros((nyears * nmonths, len(idx_dsf)))
    ppt_grw_all = np.zeros((nyears * nmonths, len(idx_grw)))
    ppt_saw_all =  np.zeros((nyears * nmonths, len(idx_saw)))

    nyear = 0
    cnt = 0
    for year in np.arange(start_yr, end_yr):
        print(year)
        fdir = "outputs"
        fname = os.path.join(fdir, "cable_out_%d.nc" % (year))
        ds = xr.open_dataset(fname)

        plc_vals = ds["plc"][:,0,:,:].values
        et_vals = ds["Evap"][:,:,:].values
        ppt_vals = ds["Rainf"][:,:,:].values

        idx = nyear + cnt



        plc_rf = np.zeros((12,len(idx_rf)))
        et_rf = np.zeros((12,len(idx_rf)))
        ppt_rf = np.zeros((12,len(idx_rf)))
        for i in range(len(idx_rf)):
            (row, col) = idx_rf[i]
            plc_rf[:,i] = plc_vals[:,row,col]
            et_rf[:,i] = et_vals[:,row,col]
            ppt_rf[:,i] = ppt_vals[:,row,col]
        plc_rf_all[idx:(idx+12),:] = plc_rf
        et_rf_all[idx:(idx+12),:] = et_rf
        ppt_rf_all[idx:(idx+12),:] = ppt_rf

        plc_wsf = np.zeros((12,len(idx_wsf)))
        et_wsf = np.zeros((12,len(idx_wsf)))
        ppt_wsf = np.zeros((12,len(idx_wsf)))
        for i in range(len(idx_wsf)):
            (row, col) = idx_wsf[i]
            plc_wsf[:,i] = plc_vals[:,row,col]
            et_wsf[:,i] = et_vals[:,row,col]
            ppt_wsf[:,i] = ppt_vals[:,row,col]
        plc_wsf_all[idx:(idx+12),:] = plc_wsf
        et_wsf_all[idx:(idx+12),:] = et_wsf
        ppt_wsf_all[idx:(idx+12),:] = ppt_wsf

        plc_dsf = np.zeros((12,len(idx_dsf)))
        et_dsf = np.zeros((12,len(idx_dsf)))
        ppt_dsf = np.zeros((12,len(idx_dsf)))
        for i in range(len(idx_dsf)):
            (row, col) = idx_dsf[i]
            plc_dsf[:,i] = plc_vals[:,row,col]
            et_dsf[:,i] = et_vals[:,row,col]
            ppt_dsf[:,i] = ppt_vals[:,row,col]
        plc_dsf_all[idx:(idx+12),:] = plc_dsf
        et_dsf_all[idx:(idx+12),:] = et_dsf
        ppt_dsf_all[idx:(idx+12),:] = ppt_dsf

        plc_grw = np.zeros((12,len(idx_grw)))
        et_grw = np.zeros((12,len(idx_grw)))
        ppt_grw = np.zeros((12,len(idx_grw)))
        for i in range(len(idx_grw)):
            (row, col) = idx_grw[i]
            plc_grw[:,i] = plc_vals[:,row,col]
            et_grw[:,i] = et_vals[:,row,col]
            ppt_grw[:,i] = ppt_vals[:,row,col]
        plc_grw_all[idx:(idx+12),:] = plc_grw
        et_grw_all[idx:(idx+12),:] = et_grw
        ppt_grw_all[idx:(idx+12),:] = ppt_grw

        plc_saw = np.zeros((12,len(idx_saw)))
        et_saw = np.zeros((12,len(idx_saw)))
        ppt_saw = np.zeros((12,len(idx_saw)))
        for i in range(len(idx_saw)):
            (row, col) = idx_saw[i]
            plc_saw[:,i] = plc_vals[:,row,col]
            et_saw[:,i] = et_vals[:,row,col]
            ppt_saw[:,i] = ppt_vals[:,row,col]
        plc_saw_all[idx:(idx+12),:] = plc_saw
        et_saw_all[idx:(idx+12),:] = et_saw
        ppt_saw_all[idx:(idx+12),:] = ppt_saw

        nyear += 1
        cnt += 12

    sec_2_day = 86400.0
    cnt = 0
    for year in np.arange(start_yr, end_yr):
        for month in range(1,12+1):
            days_in_month = monthrange(year, month)[1]
            conv = sec_2_day * days_in_month
            ppt_rf_all[cnt,:] *= conv
            et_rf_all[cnt,:] *= conv
            ppt_wsf_all[cnt,:] *= conv
            et_wsf_all[cnt,:] *= conv
            ppt_dsf_all[cnt,:] *= conv
            et_dsf_all[cnt,:] *= conv
            ppt_saw_all[cnt,:] *= conv
            et_saw_all[cnt,:] *= conv

            cnt += 1

    from scipy import stats
    a = ppt_wsf_all[:,0] - et_wsf_all[:,0]
    b = plc_wsf_all[:,0]
    r, pvalue = stats.pearsonr(a, b)
    print("r: %f; p-value: %f" % (r, pvalue))

    fig, ax1 = plt.subplots()

    sec_2_day = 86400.0

    ax1.plot(ppt_wsf_all[:,0] - et_wsf_all[:,0], color="red")
    ax1.set_ylabel("PPT-AET")

    ax2 = ax1.twinx()

    ax2.plot(plc_wsf_all[:,0], color="green")
    ax2.set_ylabel("PLC")
    plt.show()


if __name__ == "__main__":

    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    main(plot_dir)

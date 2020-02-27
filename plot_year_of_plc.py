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

    """
    plc_when_saw = np.zeros(len(idx_wsf))
    for i in range(len(idx_wsf)):
        cnt = 0
        min_plc = -99.
        for year in np.arange(start_yr, end_yr):
            for month in range(1,12+1):

                if plc_saw_all[cnt,i] > min_plc:

                    plc_when_saw[i] = plc_saw_all[cnt,i]#year #scnt
                    min_plc = plc_saw_all[cnt,i]

                cnt += 1
    """
    import pandas as pd
    data = {}
    data["RF"] = plc_rf_all.flatten()
    data["WSF"] = plc_wsf_all.flatten()
    df = pd.DataFrame(data).T
    print(df.summary())
    sys.exit()
    #data = [plc_rf_all.flatten(), plc_wsf_all.flatten(), plc_dsf_all.flatten(),\
    #        plc_grw_all.flatten(), plc_saw_all.flatten()]
    import seaborn as sns

    sns.boxplot(data=data, whis=np.inf, width=.18)
    sns.swarmplot(data=data, size=6, edgecolor="black", linewidth=.9)
    #plt.boxplot(data)
    plt.show()


if __name__ == "__main__":

    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    main(plot_dir)

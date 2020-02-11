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

    nyear = 0
    cnt = 0
    for year in np.arange(start_yr, end_yr):
        print(year)
        fdir = "outputs"
        fname = os.path.join(fdir, "cable_out_%d.nc" % (year))
        ds = xr.open_dataset(fname)
        plc_vals = ds["plc"][:,0,:,:].values

        idx = nyear + cnt

        plc_rf = np.zeros((12,len(idx_rf)))
        for i in range(len(idx_rf)):
            (row, col) = idx_rf[i]
            plc_rf[:,i] = plc_vals[:,row,col]
        plc_rf_all[idx:(idx+12),:] = plc_rf

        plc_wsf = np.zeros((12,len(idx_wsf)))
        for i in range(len(idx_wsf)):
            (row, col) = idx_wsf[i]
            plc_wsf[:,i] = plc_vals[:,row,col]
        plc_wsf_all[idx:(idx+12),:] = plc_wsf

        plc_dsf = np.zeros((12,len(idx_dsf)))
        for i in range(len(idx_dsf)):
            (row, col) = idx_dsf[i]
            plc_dsf[:,i] = plc_vals[:,row,col]
        plc_dsf_all[idx:(idx+12),:] = plc_dsf

        plc_grw = np.zeros((12,len(idx_grw)))
        for i in range(len(idx_grw)):
            (row, col) = idx_grw[i]
            plc_grw[:,i] = plc_vals[:,row,col]
        plc_grw_all[idx:(idx+12),:] = plc_grw

        plc_saw = np.zeros((12,len(idx_saw)))
        for i in range(len(idx_saw)):
            (row, col) = idx_saw[i]
            plc_saw[:,i] = plc_vals[:,row,col]
        plc_saw_all[idx:(idx+12),:] = plc_saw


        nyear += 1
        cnt += 12


    #from matplotlib.pyplot import cm
    #colours = cm.Set2(np.linspace(0, 1, 5))
    #colours = cm.get_cmap('Set2')


    import seaborn as sns
    sns.set_style("ticks")
    colours = sns.color_palette("Set2", 8)


    fig = plt.figure(figsize=(6,9))
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.05)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"

    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['font.size'] = 14
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    ax1 = fig.add_subplot(511)
    ax2 = fig.add_subplot(512)
    ax3 = fig.add_subplot(513)
    ax4 = fig.add_subplot(514)
    ax5 = fig.add_subplot(515)


    data = plc_rf_all.reshape(nyears * nmonths * len(idx_rf))
    data = data[~np.isnan(data)]
    index = np.arange(len(data))
    df = pd.DataFrame(data=data, index=index, columns=["PLC"], dtype='float')
    sns.distplot(df.PLC, ax=ax1, norm_hist=True, kde_kws={"label": "KDE"})

    data = plc_wsf_all.reshape(nyears * nmonths * len(idx_wsf))
    data = data[~np.isnan(data)]
    index = np.arange(len(data))
    df = pd.DataFrame(data=data, index=index, columns=["PLC"], dtype='float')
    sns.distplot(df.PLC, ax=ax2, norm_hist=True)

    data = plc_dsf_all.reshape(nyears * nmonths * len(idx_dsf))
    data = data[~np.isnan(data)]
    index = np.arange(len(data))
    df = pd.DataFrame(data=data, index=index, columns=["PLC"], dtype='float')
    sns.distplot(df.PLC, ax=ax3, norm_hist=True)

    data = plc_grw_all.reshape(nyears * nmonths * len(idx_grw))
    data = data[~np.isnan(data)]
    index = np.arange(len(data))
    df = pd.DataFrame(data=data, index=index, columns=["PLC"], dtype='float')
    sns.distplot(df.PLC, ax=ax4, norm_hist=True)

    data = plc_saw_all.reshape(nyears * nmonths * len(idx_saw))
    data = data[~np.isnan(data)]
    index = np.arange(len(data))
    df = pd.DataFrame(data=data, index=index, columns=["PLC"], dtype='float')
    sns.distplot(df.PLC, ax=ax5, norm_hist=True)

    ax1.set_xlim(-5, 90)
    ax2.set_xlim(-5, 90)
    ax3.set_xlim(-5, 90)
    ax4.set_xlim(-5, 90)
    ax5.set_xlim(-5, 90)

    #ax1.set_ylim(0.0, 0.45)
    #ax2.set_ylim(0.0, 0.45)
    #ax3.set_ylim(0.0, 0.45)
    #ax4.set_ylim(0.0, 0.45)
    #ax5.set_ylim(0.0, 0.45)
    from matplotlib.ticker import MaxNLocator
    ax1.yaxis.set_major_locator(MaxNLocator(3))
    ax1.xaxis.set_major_locator(MaxNLocator(5))
    ax1.tick_params(direction='in', length=4)

    ax2.yaxis.set_major_locator(MaxNLocator(3))
    ax2.xaxis.set_major_locator(MaxNLocator(5))
    ax2.tick_params(direction='in', length=4)

    ax3.yaxis.set_major_locator(MaxNLocator(3))
    ax3.xaxis.set_major_locator(MaxNLocator(5))
    ax3.tick_params(direction='in', length=4)

    ax4.yaxis.set_major_locator(MaxNLocator(3))
    ax4.xaxis.set_major_locator(MaxNLocator(5))
    ax4.tick_params(direction='in', length=4)

    ax5.yaxis.set_major_locator(MaxNLocator(3))
    ax5.xaxis.set_major_locator(MaxNLocator(5))
    ax5.tick_params(direction='in', length=4)

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax4.get_xticklabels(), visible=False)

    ax1.set_xlabel(" ")
    ax2.set_xlabel(" ")
    ax3.set_xlabel(" ")
    ax4.set_xlabel(" ")

    odir = "plots"
    plt.savefig(os.path.join(odir, "plc_timeseries_hist.pdf"),
                bbox_inches='tight', pad_inches=0.1)

    plt.show()




if __name__ == "__main__":

    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    main(plot_dir)

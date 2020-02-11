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

    plc_rf_sigma = np.zeros(0)
    plc_wsf_sigma = np.zeros(0)
    plc_dsf_sigma = np.zeros(0)
    plc_grw_sigma = np.zeros(0)
    plc_saw_sigma =  np.zeros(0)

    plc_rf_min_all = np.zeros(0)
    plc_wsf_min_all = np.zeros(0)
    plc_dsf_min_all = np.zeros(0)
    plc_grw_min_all = np.zeros(0)
    plc_saw_min_all = np.zeros(0)

    plc_rf_max_all = np.zeros(0)
    plc_wsf_max_all = np.zeros(0)
    plc_dsf_max_all = np.zeros(0)
    plc_grw_max_all = np.zeros(0)
    plc_saw_max_all = np.zeros(0)

    sw_rf_all = np.zeros(0)
    sw_wsf_all = np.zeros(0)
    sw_dsf_all = np.zeros(0)
    sw_grw_all = np.zeros(0)
    sw_saw_all = np.zeros(0)

    start_yr = 2000
    end_yr = 2010

    for year in np.arange(start_yr, end_yr):
    #for year in np.arange(2000, 2001):
    #for year in np.arange(2000, 2002):

        #fdir = "/Users/mdekauwe/Desktop/outputs"
        fdir = "outputs"

        fname = os.path.join(fdir, "cable_out_%d.nc" % (year))
        ds = xr.open_dataset(fname)

        iveg = ds["iveg"][:,:].values
        plc_vals = ds["plc"][:,0,:,:].values

        SoilMoist1 = ds["SoilMoist"][:,0,:,:].values * zse[0]
        SoilMoist2 = ds["SoilMoist"][:,1,:,:].values * zse[1]
        SoilMoist3 = ds["SoilMoist"][:,2,:,:].values * zse[2]
        SoilMoist4 = ds["SoilMoist"][:,3,:,:].values * zse[3]
        SoilMoist5 = ds["SoilMoist"][:,4,:,:].values * zse[4]
        SoilMoist6 = ds["SoilMoist"][:,5,:,:].values * zse[5]
        sw = (SoilMoist1 + SoilMoist2 + SoilMoist3 + \
                SoilMoist4 + SoilMoist5 + SoilMoist6 ) / np.sum(zse)


        idx_rf = np.argwhere(iveg == 18.0)
        idx_wsf = np.argwhere(iveg == 19.0)
        idx_dsf = np.argwhere(iveg == 20.0)
        idx_grw = np.argwhere(iveg == 21.0)
        idx_saw = np.argwhere(iveg == 22.0)

        plc_rf = np.zeros((12,len(idx_rf)))
        sw_rf = np.zeros((12,len(idx_rf)))
        for i in range(len(idx_rf)):
            (row, col) = idx_rf[i]
            plc_rf[:,i] = plc_vals[:,row,col]
            sw_rf[:,i] = sw[:,row,col]

        plc_wsf = np.zeros((12,len(idx_wsf)))
        sw_wsf = np.zeros((12,len(idx_wsf)))
        for i in range(len(idx_wsf)):
            (row, col) = idx_wsf[i]
            plc_wsf[:,i] = plc_vals[:,row,col]
            sw_wsf[:,i] = sw[:,row,col]

        plc_dsf = np.zeros((12,len(idx_dsf)))
        sw_dsf = np.zeros((12,len(idx_dsf)))
        for i in range(len(idx_dsf)):
            (row, col) = idx_dsf[i]
            plc_dsf[:,i] = plc_vals[:,row,col]
            sw_dsf[:,i] = sw[:,row,col]

        plc_grw = np.zeros((12,len(idx_grw)))
        sw_grw = np.zeros((12,len(idx_grw)))
        for i in range(len(idx_grw)):
            (row, col) = idx_grw[i]
            plc_grw[:,i] = plc_vals[:,row,col]
            sw_grw[:,i] = sw[:,row,col]

        plc_saw = np.zeros((12,len(idx_saw)))
        sw_saw = np.zeros((12,len(idx_saw)))
        for i in range(len(idx_saw)):
            (row, col) = idx_saw[i]
            plc_saw[:,i] = plc_vals[:,row,col]
            sw_saw[:,i] = sw[:,row,col]


        plc_rf_sig = np.std(plc_rf, axis=1)
        plc_wsf_sig = np.std(plc_wsf, axis=1)
        plc_dsf_sig = np.std(plc_dsf, axis=1)
        plc_grw_sig = np.std(plc_grw, axis=1)
        plc_saw_sig = np.std(plc_saw, axis=1)

        plc_rf_max = np.amax(plc_rf, axis=1)
        plc_wsf_max = np.amax(plc_wsf, axis=1)
        plc_dsf_max = np.amax(plc_dsf, axis=1)
        plc_grw_max = np.amax(plc_grw, axis=1)
        plc_saw_max = np.amax(plc_saw, axis=1)

        plc_rf_min = np.amin(plc_rf, axis=1)
        plc_wsf_min = np.amin(plc_wsf, axis=1)
        plc_dsf_min = np.amin(plc_dsf, axis=1)
        plc_grw_min = np.amin(plc_grw, axis=1)
        plc_saw_min = np.amin(plc_saw, axis=1)


        plc_rf_mu = np.nanmean(plc_rf, axis=1)
        plc_wsf_mu = np.nanmean(plc_wsf, axis=1)
        plc_dsf_mu = np.nanmean(plc_dsf, axis=1)
        plc_grw_mu = np.nanmean(plc_grw, axis=1)
        plc_saw_mu = np.nanmean(plc_saw, axis=1)

        sw_rf = np.nanmean(sw_rf, axis=1)
        sw_wsf = np.nanmean(sw_wsf, axis=1)
        sw_dsf = np.nanmean(sw_dsf, axis=1)
        sw_grw = np.nanmean(sw_grw, axis=1)
        sw_saw = np.nanmean(sw_saw, axis=1)

        plc_rf_all = np.append(plc_rf_all, plc_rf_mu)
        plc_wsf_all = np.append(plc_wsf_all, plc_wsf_mu)
        plc_dsf_all = np.append(plc_dsf_all, plc_dsf_mu)
        plc_grw_all = np.append(plc_grw_all, plc_grw_mu)
        plc_saw_all = np.append(plc_saw_all, plc_saw_mu)

        plc_rf_sigma = np.append(plc_rf_sigma, plc_rf_sig)
        plc_wsf_sigma = np.append(plc_wsf_sigma, plc_wsf_sig)
        plc_dsf_sigma = np.append(plc_dsf_sigma, plc_dsf_sig)
        plc_grw_sigma = np.append(plc_grw_sigma, plc_grw_sig)
        plc_saw_sigma = np.append(plc_saw_sigma, plc_saw_sig)

        plc_rf_min_all = np.append(plc_rf_min_all, plc_rf_min)
        plc_wsf_min_all = np.append(plc_wsf_min_all, plc_wsf_min)
        plc_dsf_min_all = np.append(plc_dsf_min_all, plc_dsf_min)
        plc_grw_min_all = np.append(plc_grw_min_all, plc_grw_min)
        plc_saw_min_all = np.append(plc_saw_min_all, plc_saw_min)

        plc_rf_max_all = np.append(plc_rf_max_all, plc_rf_max)
        plc_wsf_max_all = np.append(plc_wsf_max_all, plc_wsf_max)
        plc_dsf_max_all = np.append(plc_dsf_max_all, plc_dsf_max)
        plc_grw_max_all = np.append(plc_grw_max_all, plc_grw_max)
        plc_saw_max_all = np.append(plc_saw_max_all, plc_saw_max)


        sw_rf_all = np.append(sw_rf_all, sw_rf)
        sw_wsf_all = np.append(sw_wsf_all, sw_wsf)
        sw_dsf_all = np.append(sw_dsf_all, sw_dsf)
        sw_grw_all = np.append(sw_grw_all, sw_grw)
        sw_saw_all = np.append(sw_saw_all, sw_saw)

        #plt.plot(plc_rf, label="RF")
        #plt.plot(plc_wsf, label="WSF")
        #plt.plot(plc_dsf, label="DSF")
        #plt.plot(plc_grw, label="GRW")
        #plt.plot(plc_saw, label="SAW")

        #plt.scatter(sw_rf, plc_rf, label="RF")
        #plt.plot(sw_wsf, label="WSF")
        #plt.plot(sw_dsf, label="DSF")
        #plt.plot(sw_grw, label="GRW")
        #plt.plot(sw_saw, label="SAW")

    #plt.scatter(sw_rf_all, plc_rf_all, label="RF")
    #plt.scatter(sw_wsf_all, plc_wsf_all, label="WSF")
    #plt.scatter(sw_dsf_all, plc_dsf_all, label="DSF")
    #plt.scatter(sw_grw_all, plc_grw_all, label="GRW")
    #plt.scatter(sw_saw_all, plc_saw_all, label="SAW")
    periods = (end_yr - start_yr) * 12
    dates = pd.date_range('01/01/%d' % (start_yr), periods=periods, freq ='M')

    #from matplotlib.pyplot import cm
    #colours = cm.Set2(np.linspace(0, 1, 5))
    #colours = cm.get_cmap('Set2')


    import seaborn as sns
    sns.set_style("ticks")
    colours = sns.color_palette("Set2", 8)


    fig = plt.figure(figsize=(9,6))
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

    ax = fig.add_subplot(111)

    ax.plot(dates, plc_rf_all, label="RF", color=colours[0], lw=2)
    #ax.plot(dates, plc_wsf_all, label="WSF", color=colours[1], lw=2)
    #ax.plot(dates, plc_dsf_all, label="DSF", color=colours[2], lw=2)
    #ax.plot(dates, plc_grw_all, label="GRW", color=colours[3], lw=2)
    #ax.plot(dates, plc_saw_all, label="SAW", color=colours[4], lw=2)

    ax.fill_between(dates, plc_rf_all+plc_rf_max_all, plc_rf_all-plc_rf_min_all,
                    facecolor=colours[6], alpha=0.5)
    #ax.fill_between(dates, plc_wsf_all+plc_wsf_max_all, plc_wsf_all-plc_wsf_min_all,
    #                facecolor=colours[1], alpha=0.5)
    #ax.fill_between(dates, plc_dsf_all+plc_dsf_max_all, plc_dsf_all-plc_dsf_min_all,
    #                facecolor=colours[2], alpha=0.5)
    #ax.fill_between(dates, plc_grw_all+plc_grw_max_all, plc_grw_all-plc_grw_min_all,
    #                facecolor=colours[3], alpha=0.5)
    #ax.fill_between(dates, plc_saw_all+plc_saw_max_all, plc_saw_all-plc_saw_min_all,
    #                facecolor=colours[0], alpha=0.5)

    #ax.fill_between(dates, plc_rf_all+plc_rf_sigma, plc_rf_all-plc_rf_sigma,
    #                facecolor=colours[6], alpha=0.5)
    #ax.fill_between(dates, plc_wsf_all+plc_wsf_sigma, plc_wsf_all-plc_wsf_sigma,
    #                facecolor=colours[1], alpha=0.5)
    #ax.fill_between(dates, plc_dsf_all+plc_dsf_sigma, plc_dsf_all-plc_dsf_sigma,
    #                facecolor=colours[2], alpha=0.5)
    #ax.fill_between(dates, plc_grw_all+plc_grw_sigma, plc_grw_all-plc_grw_sigma,
    #                facecolor=colours[3], alpha=0.5)
    #ax.fill_between(dates, plc_saw_all+plc_saw_sigma, plc_saw_all-plc_saw_sigma,
    #                facecolor=colours[0], alpha=0.5)


    ax.axhline(y=88.0, ls="--", lw=2, color="black", label="$\Psi$$_{crit}$")
    ax.set_ylim(-5, 90)

    ax.set_ylabel("Loss of hydraulic conductivity (%)")
    ax.legend(numpoints=1, loc=(0.01, 0.65), ncol=1, frameon=False)

    import datetime
    ax.set_xlim([datetime.date(2000,7,1), datetime.date(2010, 1, 1)])

    odir = "plots"
    plt.savefig(os.path.join(odir, "plc_timeseries_test.pdf"),,
                bbox_inches='tight', pad_inches=0.1)

    plt.show()




if __name__ == "__main__":

    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    main(plot_dir)

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

    sw_wsf_all = np.zeros(0)
    sw_wsf_sigma = np.zeros(0)

    sw_wsf_allh = np.zeros(0)
    sw_wsf_sigmah = np.zeros(0)

    start_yr = 2000
    end_yr = 2010

    for year in np.arange(start_yr, end_yr):
    #for year in np.arange(2000, 2001):
    #for year in np.arange(2000, 2002):

        #fdir = "/Users/mdekauwe/Desktop/outputs"
        #fdir = "outputs"
        fdir = "outputs2"

        fname = os.path.join(fdir, "cable_out_%d.nc" % (year))
        ds = xr.open_dataset(fname)


        fdir2 = "outputs"
        fname2 = os.path.join(fdir2, "cable_out_%d.nc" % (year))
        ds2 = xr.open_dataset(fname2)
        iveg = ds2["iveg"][:,:].values


        SoilMoist1 = ds["SoilMoist"][:,0,:,:].values * zse[0]
        SoilMoist2 = ds["SoilMoist"][:,1,:,:].values * zse[1]
        SoilMoist3 = ds["SoilMoist"][:,2,:,:].values * zse[2]
        SoilMoist4 = ds["SoilMoist"][:,3,:,:].values * zse[3]
        SoilMoist5 = ds["SoilMoist"][:,4,:,:].values * zse[4]
        SoilMoist6 = ds["SoilMoist"][:,5,:,:].values * zse[5]
        sw = (SoilMoist1 + SoilMoist2 + SoilMoist3 + \
                SoilMoist4 + SoilMoist5 + SoilMoist6 ) / np.sum(zse)


        SoilMoist1h = ds2["SoilMoist"][:,0,:,:].values * zse[0]
        SoilMoist2h = ds2["SoilMoist"][:,1,:,:].values * zse[1]
        SoilMoist3h = ds2["SoilMoist"][:,2,:,:].values * zse[2]
        SoilMoist4h = ds2["SoilMoist"][:,3,:,:].values * zse[3]
        SoilMoist5h = ds2["SoilMoist"][:,4,:,:].values * zse[4]
        SoilMoist6h = ds2["SoilMoist"][:,5,:,:].values * zse[5]
        swh = (SoilMoist1h + SoilMoist2h + SoilMoist3h + \
                SoilMoist4h + SoilMoist5h + SoilMoist6h ) / np.sum(zse)

        idx_wsf = np.argwhere(iveg == 19.0)

        sw_wsfh = np.zeros((12,len(idx_wsf)))
        sw_wsf = np.zeros((12,len(idx_wsf)))
        for i in range(len(idx_wsf)):
            (row, col) = idx_wsf[i]

            sw_wsf[:,i] = sw[:,row,col]
            sw_wsfh[:,i] = swh[:,row,col]


        sw_wsf_mu = np.mean(sw_wsf, axis=1)
        sw_wsf_sig = np.std(sw_wsf, axis=1)

        sw_wsf_all = np.append(sw_wsf_all, sw_wsf_mu)
        sw_wsf_sigma = np.append(sw_wsf_sigma, sw_wsf_sig)

        sw_wsf_muh = np.mean(sw_wsfh, axis=1)
        sw_wsf_sigh = np.std(sw_wsfh, axis=1)

        sw_wsf_allh = np.append(sw_wsf_allh, sw_wsf_muh)
        sw_wsf_sigmah = np.append(sw_wsf_sigmah, sw_wsf_sigh)

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


    #ax.plot(dates, plc_wsf_all, label="WSF", color=colours[1], lw=2)
    ax.plot(dates, sw_wsf_all, label="Control", color=colours[1], lw=2)
    ax.fill_between(dates, sw_wsf_all+sw_wsf_sigma, sw_wsf_all-sw_wsf_sigma,
                    facecolor=colours[1], alpha=0.5)


    ax.plot(dates, sw_wsf_allh, label="Hydraulics", color=colours[2], lw=2)
    ax.fill_between(dates, sw_wsf_allh+sw_wsf_sigmah, sw_wsf_allh-sw_wsf_sigmah,
                    facecolor=colours[2], alpha=0.5)

    #ax.axhline(y=88.0, ls="--", lw=2, color="black", label="$\Psi$$_{crit}$")
    #ax.set_ylim(-5, 90)

    ax.set_ylabel("Volumetric soil water content (m$^{3}$ m$^{-3}$)")
    ax.legend(numpoints=1, loc="best", ncol=1, frameon=False)

    import datetime
    ax.set_xlim([datetime.date(2000,7,1), datetime.date(2010, 1, 1)])

    odir = "/Users/mdekauwe/Dropbox/Drought_risk_paper/figures/figs"
    plt.savefig(os.path.join(odir, "vwc_WSF_timeseries_control_vs_hydraulics.pdf"),
                bbox_inches='tight', pad_inches=0.1)

    plt.show()




if __name__ == "__main__":

    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    main(plot_dir)

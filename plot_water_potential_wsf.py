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

    lwp_wsf_all = np.zeros(0)
    lwp_wsf_sigma = np.zeros(0)

    swp_wsf_all = np.zeros(0)
    swp_wsf_sigma = np.zeros(0)

    xwp_wsf_all = np.zeros(0)
    xwp_wsf_sigma = np.zeros(0)

    fw_wsf_all = np.zeros(0)
    fw_wsf_sigma = np.zeros(0)

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

        psi_leaf = ds["psi_leaf"][:,0,:,:].values
        psi_stem = ds["psi_stem"][:,0,:,:].values
        psi_soil = ds["weighted_psi_soil"][:,0,:,:].values

        idx_wsf = np.argwhere(iveg == 19.0)


        fdir = "outputs2"
        fname = os.path.join(fdir, "cable_out_%d.nc" % (year))
        ds2 = xr.open_dataset(fname)
        fwsoil = ds2["Fwsoil"][:,:,:].values


        lwp_wsf = np.zeros((12,len(idx_wsf)))
        swp_wsf = np.zeros((12,len(idx_wsf)))
        xwp_wsf = np.zeros((12,len(idx_wsf)))
        fw_wsf = np.zeros((12,len(idx_wsf)))
        for i in range(len(idx_wsf)):
            (row, col) = idx_wsf[i]

            lwp_wsf[:,i] = psi_leaf[:,row,col]
            swp_wsf[:,i] = psi_soil[:,row,col]
            xwp_wsf[:,i] = psi_stem[:,row,col]
            fw_wsf[:,i] = fwsoil[:,row,col]

        fw_wsf_mu = np.mean(fw_wsf, axis=1)
        fw_wsf_sig = np.std(fw_wsf, axis=1)

        fw_wsf_all = np.append(fw_wsf_all, fw_wsf_mu)
        fw_wsf_sigma = np.append(fw_wsf_sigma, fw_wsf_sig)

        lwp_wsf_mu = np.mean(lwp_wsf, axis=1)
        lwp_wsf_sig = np.std(lwp_wsf, axis=1)

        swp_wsf_mu = np.mean(swp_wsf, axis=1)
        swp_wsf_sig = np.std(swp_wsf, axis=1)

        xwp_wsf_mu = np.mean(xwp_wsf, axis=1)
        xwp_wsf_sig = np.std(xwp_wsf, axis=1)

        lwp_wsf_all = np.append(lwp_wsf_all, lwp_wsf_mu)
        lwp_wsf_sigma = np.append(lwp_wsf_sigma, lwp_wsf_sig)

        swp_wsf_all = np.append(swp_wsf_all, swp_wsf_mu)
        swp_wsf_sigma = np.append(swp_wsf_sigma, swp_wsf_sig)

        xwp_wsf_all = np.append(xwp_wsf_all, xwp_wsf_mu)
        xwp_wsf_sigma = np.append(xwp_wsf_sigma, xwp_wsf_sig)



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


    fig = plt.figure(figsize=(9,10))
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.2)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"

    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['font.size'] = 14
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.plot(dates, fw_wsf_all, label="WSF", color=colours[2], lw=2)

    ax1.fill_between(dates, fw_wsf_all+fw_wsf_sigma, fw_wsf_all-fw_wsf_sigma,
                    facecolor=colours[2], alpha=0.5)

    ax2.plot(dates, lwp_wsf_all, label="$\Psi$$_{l}$", color=colours[0], lw=2)
    #ax.fill_between(dates, lwp_wsf_all+lwp_wsf_sigma, lwp_wsf_all-lwp_wsf_sigma,
    #                facecolor=colours[0], alpha=0.5)

    ax2.plot(dates, xwp_wsf_all, label="$\Psi$$_{x}$", color=colours[1], lw=2)
    #ax.fill_between(dates, xwp_wsf_all+xwp_wsf_sigma, xwp_wsf_all-xwp_wsf_sigma,
    #                facecolor=colours[1], alpha=0.5)

    ax2.plot(dates, swp_wsf_all, label="$\Psi$$_{s,weight}$", color=colours[2], lw=2)
    #ax.fill_between(dates, swp_wsf_all+swp_wsf_sigma, swp_wsf_all-swp_wsf_sigma,
    #                facecolor=colours[2], alpha=0.5)

    ax1.set_ylabel(r"$\beta$ (-)")
    ax2.set_ylabel("Water potential (MPa)")
    ax2.legend(numpoints=1, loc="best", ncol=1, frameon=False)

    import datetime
    ax1.set_xlim([datetime.date(2000,7,1), datetime.date(2010, 1, 1)])
    ax2.set_xlim([datetime.date(2000,7,1), datetime.date(2010, 1, 1)])
    plt.setp(ax1.get_xticklabels(), visible=False)

    odir = "/Users/mdekauwe/Dropbox/Drought_risk_paper/figures/figs"
    plt.savefig(os.path.join(odir, "Fwsoil_vs_water_potentials_WSF.pdf"),
                bbox_inches='tight', pad_inches=0.1)

    plt.show()




if __name__ == "__main__":

    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    main(plot_dir)

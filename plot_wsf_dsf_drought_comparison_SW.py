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
import string

def md_drought(plot_dir):

    # layer thickness
    zse = np.array([.022, .058, .154, .409, 1.085, 2.872])

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
        """
        sw = (SoilMoist1 + SoilMoist2 + \
              SoilMoist3 + SoilMoist4) / np.sum(zse[0:4])

        idx = nyear + cnt

        sw_rf = np.zeros((12,len(idx_rf)))
        for i in range(len(idx_rf)):
            (row, col) = idx_rf[i]
            sw_rf[:,i] = sw[:,row,col]
        sw_rf_all[idx:(idx+12),:] = sw_rf

        sw_wsf = np.zeros((12,len(idx_wsf)))
        for i in range(len(idx_wsf)):
            (row, col) = idx_wsf[i]
            sw_wsf[:,i] = sw[:,row,col]
        sw_wsf_all[idx:(idx+12),:] = sw_wsf

        sw_dsf = np.zeros((12,len(idx_dsf)))
        for i in range(len(idx_dsf)):
            (row, col) = idx_dsf[i]
            sw_dsf[:,i] = sw[:,row,col]
        sw_dsf_all[idx:(idx+12),:] = sw_dsf

        nyear += 1
        cnt += 12

    sw_rf_all = sw_rf_all.reshape(nyears * nmonths * len(idx_rf))
    sw_wsf_all = sw_wsf_all.reshape(nyears * nmonths * len(idx_wsf))
    sw_dsf_all = sw_dsf_all.reshape(nyears * nmonths * len(idx_dsf))

    return (sw_rf_all, sw_wsf_all, sw_dsf_all)

def cd_drought(plot_dir):

    # layer thickness
    zse = np.array([.022, .058, .154, .409, 1.085, 2.872])

    sw_rf_all = np.zeros(0)
    sw_wsf_all = np.zeros(0)
    sw_dsf_all = np.zeros(0)
    sw_grw_all = np.zeros(0)
    sw_saw_all = np.zeros(0)

    start_yr = 2017
    end_yr = 2020
    nyears = (end_yr - start_yr) + 1
    nmonths = 12

    fdir = "../current/outputs"
    fname = os.path.join(fdir, "cable_out_%d.nc" % (start_yr))
    ds = xr.open_dataset(fname)
    iveg = ds["iveg"][:,:].values
    idx_rf = np.argwhere(iveg == 18.0)
    idx_wsf = np.argwhere(iveg == 19.0)
    idx_dsf = np.argwhere(iveg == 20.0)
    idx_grw = np.argwhere(iveg == 21.0)
    idx_saw = np.argwhere(iveg == 22.0)

    sw_rf_all = np.zeros((nyears * nmonths, len(idx_rf)))
    sw_wsf_all = np.zeros((nyears * nmonths, len(idx_wsf)))
    sw_dsf_all = np.zeros((nyears * nmonths, len(idx_dsf)))
    sw_grw_all = np.zeros((nyears * nmonths, len(idx_grw)))
    sw_saw_all =  np.zeros((nyears * nmonths, len(idx_saw)))

    nyear = 0
    cnt = 0
    for year in np.arange(start_yr, end_yr):
        print(year)
        fname = os.path.join(fdir, "cable_out_%d.nc" % (year))
        ds = xr.open_dataset(fname)
        plc_vals = ds["plc"][:,0,:,:].values


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
        """
        sw = (SoilMoist1 + SoilMoist2 + \
              SoilMoist3 + SoilMoist4) / np.sum(zse[0:4])

        idx = nyear + cnt

        sw_rf = np.zeros((12,len(idx_rf)))
        for i in range(len(idx_rf)):
            (row, col) = idx_rf[i]
            sw_rf[:,i] = sw[:,row,col]
        sw_rf_all[idx:(idx+12),:] = sw_rf

        sw_wsf = np.zeros((12,len(idx_wsf)))
        for i in range(len(idx_wsf)):
            (row, col) = idx_wsf[i]
            sw_wsf[:,i] = sw[:,row,col]
        sw_wsf_all[idx:(idx+12),:] = sw_wsf

        sw_dsf = np.zeros((12,len(idx_dsf)))
        for i in range(len(idx_dsf)):
            (row, col) = idx_dsf[i]
            sw_dsf[:,i] = sw[:,row,col]
        sw_dsf_all[idx:(idx+12),:] = sw_dsf



        nyear += 1
        cnt += 12

    sw_rf_all = sw_rf_all.reshape(nyears * nmonths * len(idx_rf))
    sw_wsf_all = sw_wsf_all.reshape(nyears * nmonths * len(idx_wsf))
    sw_dsf_all = sw_dsf_all.reshape(nyears * nmonths * len(idx_dsf))


    return (sw_rf_all, sw_wsf_all, sw_dsf_all)


def label_generator(case='lower', start='', end=''):
    choose_type = {'lower': string.ascii_lowercase,
                   'upper': string.ascii_uppercase}
    generator = ('%s%s%s' %(start, letter, end) for letter in choose_type[case])

    return generator

if __name__ == "__main__":

    #plot_dir = "plots"
    plot_dir = "/Users/mdekauwe/Dropbox/Drought_risk_paper/figures/figs"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    (sw_rf_md, sw_wsf_md, sw_dsf_md) = md_drought(plot_dir)
    (sw_rf_cd, sw_wsf_cd, sw_dsf_cd) = cd_drought(plot_dir)

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

    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)


    index = np.arange(len(sw_rf_md))
    df = pd.DataFrame(data=sw_rf_md, index=index, columns=["SW"], dtype='float')
    sns.distplot(df.SW, ax=ax1, norm_hist=True,
                 label="Millennium drought")

    index = np.arange(len(sw_rf_cd))
    df = pd.DataFrame(data=sw_rf_cd, index=index, columns=["SW"], dtype='float')
    sns.distplot(df.SW, ax=ax1, norm_hist=True,
                 label="Current drought")

    index = np.arange(len(sw_wsf_md))
    df = pd.DataFrame(data=sw_wsf_md, index=index, columns=["SW"], dtype='float')
    sns.distplot(df.SW, ax=ax2, norm_hist=True, )

    index = np.arange(len(sw_wsf_cd))
    df = pd.DataFrame(data=sw_wsf_cd, index=index, columns=["SW"], dtype='float')
    sns.distplot(df.SW, ax=ax2, norm_hist=True, )

    index = np.arange(len(sw_dsf_md))
    df = pd.DataFrame(data=sw_dsf_md, index=index, columns=["SW"], dtype='float')
    sns.distplot(df.SW, ax=ax3, norm_hist=True, )

    index = np.arange(len(sw_dsf_cd))
    df = pd.DataFrame(data=sw_dsf_cd, index=index, columns=["SW"], dtype='float')
    sns.distplot(df.SW, ax=ax3, norm_hist=True, )

    ax1.set_xlim(0.1, 0.4)
    ax2.set_xlim(0.1, 0.4)
    ax3.set_xlim(0.1, 0.4)

    ax1.legend(numpoints=1, loc="best", ncol=1, frameon=False)

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

    ax2.set_ylabel("Probability density")
    ax3.set_xlabel(r"$\theta$ (m$^{3}$ m$^{-3}$)")

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    #plt.setp(ax3.get_xticklabels(), visible=False)
    #plt.setp(ax4.get_xticklabels(), visible=False)

    ax1.set_xlabel(" ")
    ax2.set_xlabel(" ")



    odir = "plots"
    plt.savefig(os.path.join(odir, "SW_hist_MD_vs_CD.png"), dpi=300,
                bbox_inches='tight', pad_inches=0.1)

    plt.show()

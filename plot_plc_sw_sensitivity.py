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

    labels = label_generator('lower', start="(", end=")")

    import seaborn as sns
    sns.set_style("ticks")
    colours = sns.color_palette("Set2", 8)


    fig = plt.figure(figsize=(20,9))
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.1)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"

    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['font.size'] = 20
    plt.rcParams['legend.fontsize'] = 20
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20

    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)

    ax1.scatter(sw_rf_all, plc_rf_all,  marker=".", s=0.05,
             alpha=0.05, color=colours[0], rasterized=True)
    ax2.scatter(sw_wsf_all, plc_wsf_all, marker=".", s=0.05,
             alpha=0.05, color=colours[0], rasterized=True)
    ax3.scatter(sw_dsf_all, plc_dsf_all, marker=".", s=0.05,
                alpha=0.05, color=colours[0], rasterized=True)
    ax4.scatter(sw_grw_all, plc_grw_all, marker=".", s=0.05,
                alpha=0.05, color=colours[0], rasterized=True)
    ax5.scatter(sw_saw_all, plc_saw_all, marker=".", s=0.05,
                alpha=0.05, color=colours[0], rasterized=True)

    props = dict(boxstyle='round', facecolor='white', alpha=1.0,
                 ec="white")

    fig_label = "%s %s" % (next(labels), "RF")
    ax1.text(0.8, 0.8, fig_label,
            transform=ax1.transAxes, fontsize=14, verticalalignment='top',
            bbox=props)
    fig_label = "%s %s" % (next(labels), "WSF")
    ax2.text(0.8, 0.8, fig_label,
            transform=ax2.transAxes, fontsize=14, verticalalignment='top',
            bbox=props)
    fig_label = "%s %s" % (next(labels), "DSF")
    ax3.text(0.8, 0.8, fig_label,
            transform=ax3.transAxes, fontsize=14, verticalalignment='top',
            bbox=props)
    fig_label = "%s %s" % (next(labels), "GRW")
    ax4.text(0.8, 0.8, fig_label,
            transform=ax4.transAxes, fontsize=14, verticalalignment='top',
            bbox=props)
    fig_label = "%s %s" % (next(labels), "SAW")
    ax5.text(0.8, 0.8, fig_label,
            transform=ax5.transAxes, fontsize=14, verticalalignment='top',
            bbox=props)


    ax1.axhline(y=88.0, ls="--", lw=1, color="black", label="$\Psi$$_{crit}$")
    ax2.axhline(y=88.0, ls="--", lw=1, color="black")
    ax3.axhline(y=88.0, ls="--", lw=1, color="black")
    ax4.axhline(y=88.0, ls="--", lw=1, color="black")
    ax5.axhline(y=88.0, ls="--", lw=1, color="black")

    ax1.legend(numpoints=1, loc=(0.67, 0.6), ncol=1, frameon=False)


    ax1.set_ylim(-5, 100)
    ax2.set_ylim(-5, 100)
    ax3.set_ylim(-5, 100)
    ax4.set_ylim(-5, 100)
    ax5.set_ylim(-5, 100)

    ax1.set_xlim(0.1, 0.3)
    ax2.set_xlim(0.1, 0.3)
    ax3.set_xlim(0.1, 0.3)
    ax4.set_xlim(0.1, 0.3)
    ax5.set_xlim(0.1, 0.3)

    ax1.set_ylabel("Loss of hydraulic conductivity (%)", position=(0.5, 0.0))
    ax5.set_xlabel(r"$\theta$ (m$^{3}$ m$^{-3}$)")
    #ax.legend(numpoints=1, loc=(0.01, 0.65), ncol=1, frameon=False)


    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    plt.setp(ax5.get_yticklabels(), visible=False)

    from matplotlib.ticker import MaxNLocator
    ax1.xaxis.set_major_locator(MaxNLocator(4))
    ax2.xaxis.set_major_locator(MaxNLocator(4))
    ax3.xaxis.set_major_locator(MaxNLocator(4))
    ax4.xaxis.set_major_locator(MaxNLocator(4))
    ax5.xaxis.set_major_locator(MaxNLocator(4))

    ax1.yaxis.set_major_locator(MaxNLocator(4))
    ax2.yaxis.set_major_locator(MaxNLocator(4))
    ax3.yaxis.set_major_locator(MaxNLocator(4))
    ax4.yaxis.set_major_locator(MaxNLocator(4))
    ax5.xaxis.set_major_locator(MaxNLocator(4))

    plt.savefig(os.path.join(plot_dir, "plc_vs_sw_all.pdf"), 
                bbox_inches='tight', pad_inches=0.1)

    #plt.show()


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

    main(plot_dir)

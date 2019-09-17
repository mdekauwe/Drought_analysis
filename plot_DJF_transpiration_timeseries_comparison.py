#!/usr/bin/env python
"""
Plot DJF timeseries comparison for the Millennium drought
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (25.07.2019)"
__email__ = "mdekauwe@gmail.com"

import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import sys

def main(fname_hyd, fname_ctl, fname_iveg, plot_dir):

    ds_hyd = xr.open_dataset(fname_hyd)
    ds_iveg = xr.open_dataset(fname_iveg)
    ds_ctl = xr.open_dataset(fname_ctl)

    hyd = ds_hyd.TVeg
    ctl = ds_ctl.TVeg

    hyd19 = np.where(ds_iveg["iveg"] == 19, hyd, np.nan)
    hyd20 = np.where(ds_iveg["iveg"] == 20, hyd, np.nan)
    hyd21 = np.where(ds_iveg["iveg"] == 21, hyd, np.nan)
    ctl19 = np.where(ds_iveg["iveg"] == 19, ctl, np.nan)
    ctl20 = np.where(ds_iveg["iveg"] == 20, ctl, np.nan)
    ctl21 = np.where(ds_iveg["iveg"] == 21, ctl, np.nan)

    ctl19 = np.nanmean(ctl19, axis=(1,2))
    ctl20 = np.nanmean(ctl20, axis=(1,2))
    ctl21 = np.nanmean(ctl21, axis=(1,2))
    hyd19 = np.nanmean(hyd19, axis=(1,2))
    hyd20 = np.nanmean(hyd20, axis=(1,2))
    hyd21 = np.nanmean(hyd21, axis=(1,2))



    width = 10
    height = 6
    fig = plt.figure(figsize=(width, height))
    fig.subplots_adjust(hspace=0.05)
    fig.subplots_adjust(wspace=0.02)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['font.size'] = 16
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax1.plot(hyd19, ls="-", label="HYD: WSF")
    ax1.plot(ctl19, ls="-", label="CTL: WSF")
    ax2.plot(hyd20, ls="-", label="HYD: DSF")
    ax2.plot(ctl20, ls="-", label="CTL: DSF")
    ax3.plot(hyd21, ls="-", label="HYD: SAW")
    ax3.plot(ctl21, ls="-", label="CTL: SAW")
    ax.legend(numpoints=1, ncol=1, frameon=False, loc="best")

    ax.set_xticks(np.arange(11))
    ax.set_xticklabels(['2000-1', '2001-2', '2002-3', '2003-4', \
                        '2004-5', '2005-6', '2006-7', '2007-8',\
                        '2008-9', '2009-10'], rotation=45)

    #ax.set_ylim(0, 3.5)
    ax.set_ylabel('E (mm d$^{-1}$)')
    ofname = os.path.join(plot_dir, "DJF_transpiration_timeseries_comparison.png")
    fig.savefig(ofname, dpi=150, bbox_inches='tight',
                pad_inches=0.1)

if __name__ == "__main__":

    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    fname_hyd = "outputs/djf.nc"
    #fname_ctl = "../GSWP3_SE_aus_control/outputs/djf.nc"
    fname_ctl = "../GSWP3_SE_aus_control_ebf_patch/outputs/djf.nc"
    fname_iveg = "outputs/iveg.nc"
    main(fname_hyd, fname_ctl, fname_iveg, plot_dir)

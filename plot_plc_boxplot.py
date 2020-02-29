#!/usr/bin/env python
"""
Plot DJF for each year of the Millennium drought
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (25.07.2019)"
__email__ = "mdekauwe@gmail.com"

import os
import sys
import pandas as pd
import seaborn as sns
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np


def main(plot_dir):

    theta = np.zeros(0)
    plc_rf_all = np.zeros(0)
    plc_wsf_all = np.zeros(0)
    plc_dsf_all = np.zeros(0)
    plc_grw_all = np.zeros(0)
    plc_saw_all =  np.zeros(0)


    for year in np.arange(2000, 2007):
    #for year in np.arange(2000, 2001):
    #for year in np.arange(2000, 2002):

        #fdir = "/Users/mdekauwe/Desktop/outputs"
        fdir = "outputs"

        fname = os.path.join(fdir, "cable_out_%d.nc" % (year))
        ds = xr.open_dataset(fname)

        iveg = ds["iveg"][:,:].values
        plc_vals = ds["plc"][:,0,:,:].values

        idx_rf = np.argwhere(iveg == 18.0)
        idx_wsf = np.argwhere(iveg == 19.0)
        idx_dsf = np.argwhere(iveg == 20.0)
        idx_grw = np.argwhere(iveg == 21.0)
        idx_saw = np.argwhere(iveg == 22.0)

        plc_rf = np.zeros((12,len(idx_rf)))
        for i in range(len(idx_rf)):
            (row, col) = idx_rf[i]
            plc_rf[:,i] = plc_vals[:,row,col]

        plc_wsf = np.zeros((12,len(idx_wsf)))
        for i in range(len(idx_wsf)):
            (row, col) = idx_wsf[i]
            plc_wsf[:,i] = plc_vals[:,row,col]

        plc_dsf = np.zeros((12,len(idx_dsf)))
        for i in range(len(idx_dsf)):
            (row, col) = idx_dsf[i]
            plc_dsf[:,i] = plc_vals[:,row,col]

        plc_grw = np.zeros((12,len(idx_grw)))
        for i in range(len(idx_grw)):
            (row, col) = idx_grw[i]
            plc_grw[:,i] = plc_vals[:,row,col]

        plc_saw = np.zeros((12,len(idx_saw)))
        for i in range(len(idx_saw)):
            (row, col) = idx_saw[i]
            plc_saw[:,i] = plc_vals[:,row,col]


        plc_rf = np.mean(plc_rf, axis=1)
        plc_wsf = np.mean(plc_wsf, axis=1)
        plc_dsf = np.mean(plc_dsf, axis=1)
        plc_grw = np.mean(plc_grw, axis=1)
        plc_saw = np.mean(plc_saw, axis=1)

        #plc_rf = plc_rf.flatten()
        #plc_wsf = plc_wsf.flatten()
        #plc_dsf = plc_dsf.flatten()
        #plc_grw = plc_grw.flatten()
        #plc_saw = plc_saw.flatten()


        plc_rf_all = np.append(plc_rf_all, plc_rf)
        plc_wsf_all = np.append(plc_wsf_all, plc_wsf)
        plc_dsf_all = np.append(plc_dsf_all, plc_dsf)
        plc_grw_all = np.append(plc_grw_all, plc_grw)
        plc_saw_all = np.append(plc_saw_all, plc_saw)




    data = [plc_rf_all, plc_wsf_all, plc_dsf_all,\
            plc_grw_all, plc_saw_all]
    data = [item for sublist in data for item in sublist]

    pfts = [np.repeat("RF", len(plc_rf_all)), \
            np.repeat("WSF", len(plc_wsf_all)),\
            np.repeat("DSF", len(plc_dsf_all)),\
            np.repeat("GRW", len(plc_grw_all)),\
            np.repeat("SAW", len(plc_saw_all))]
    pfts = [item for sublist in pfts for item in sublist]

    #df = pd.DataFrame(data=data,
    #                  index=np.arange(len(data)),
    #                  columns="PLC")
    s1 = pd.Series(data)
    s2 = pd.Series(pfts)
    df = pd.DataFrame(columns = ['PLC', 'PFT'])
    df['PLC'] = s1
    df['PFT'] = s2


    sns.set_style("ticks")
    sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
    set2 = sns.color_palette("Set3", 9)

    #colours = ["#97adee"]*8
    colours = ["#E4E6D7"]*8

    fig = plt.figure(figsize=(9,4.5))
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.05)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    ax = fig.add_subplot(111)

    # define outlier properties
    flierprops = dict(marker='o', markersize=3, markerfacecolor="black")

    ax = sns.boxplot(x="PFT", y="PLC", data=df,  palette="Set1",
                     flierprops=flierprops, showfliers=True, width=0.7)
    ax.set_ylabel("Loss of hydraulic conductivity (%)")
    ax.set_xlabel(" ")

    ofname = os.path.join(plot_dir, "plc_boxplot.png")
    fig.savefig(ofname, dpi=300, bbox_inches='tight',
                pad_inches=0.1)

    #plt.savefig(os.path.join(odir, "plc_boxplot.pdf"),
    #            bbox_inches='tight', pad_inches=0.1)
    #plt.show()
    

    #plt.boxplot(sw_rf_all, plc_rf_all, label="RF")
    #plt.scatter(sw_wsf_all, plc_wsf_all, label="WSF")
    #plt.scatter(sw_dsf_all, plc_dsf_all, label="DSF")
    #plt.scatter(sw_grw_all, plc_grw_all, label="GRW")
    #plt.scatter(sw_saw_all, plc_saw_all, label="SAW")

    #plt.legend()


if __name__ == "__main__":

    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    main(plot_dir)

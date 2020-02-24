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


def get_lat_lon(row, col):

    lat = -44.025 + (row * 0.05)
    lon = 111.975 + (col * 0.05)
    print(lat, lon)

def main(plot_dir):



    # layer thickness
    zse = np.array([.022, .058, .154, .409, 1.085, 2.872])


    sw_all4 = np.zeros(0)
    sw_all = np.zeros(0)
    plc_all = np.zeros(0)
    lai_all = np.zeros(0)
    psi_leaf_all = np.zeros(0)
    psi_stem_all = np.zeros(0)
    weighted_psi_soil_all = np.zeros(0)
    rain_all = np.zeros(0)

    start_yr = 2000
    end_yr = 2010

    for year in np.arange(start_yr, end_yr):
        fdir = "outputs"
        fname = os.path.join(fdir, "cable_out_%d.nc" % (year))
        ds = xr.open_dataset(fname)


        iveg = ds["iveg"][:,:].values
        plc_vals = ds["plc"][:,0,:,:].values
        lai_vals = ds["LAI"][:,:,:].values
        psi_leaf_vals = ds["psi_leaf"][:,0,:,:].values
        psi_stem_vals = ds["psi_stem"][:,0,:,:].values
        weighted_psi_soil_vals = ds["weighted_psi_soil"][:,0,:,:].values
        Rainf_vals = ds["Rainf"][:,:,:].values


        SoilMoist1 = ds["SoilMoist"][:,0,:,:].values * zse[0]
        SoilMoist2 = ds["SoilMoist"][:,1,:,:].values * zse[1]
        SoilMoist3 = ds["SoilMoist"][:,2,:,:].values * zse[2]
        SoilMoist4 = ds["SoilMoist"][:,3,:,:].values * zse[3]
        SoilMoist5 = ds["SoilMoist"][:,4,:,:].values * zse[4]
        SoilMoist6 = ds["SoilMoist"][:,5,:,:].values * zse[5]
        sw = (SoilMoist1 + SoilMoist2 + SoilMoist3 + \
                SoilMoist4 + SoilMoist5 + SoilMoist6 ) / np.sum(zse)

        sw4 = (SoilMoist1 + SoilMoist2 + SoilMoist3 + \
                SoilMoist4) / np.sum(zse[0:4])

        idx_rf = np.argwhere(iveg == 18.0)
        idx_wsf = np.argwhere(iveg == 19.0)
        idx_dsf = np.argwhere(iveg == 20.0)
        idx_grw = np.argwhere(iveg == 21.0)
        idx_saw = np.argwhere(iveg == 22.0)

        # pick PFT
        idx = idx_grw
        plc_pix = np.zeros((12,len(idx)))
        sw_pix = np.zeros((12,len(idx)))
        lai_pix = np.zeros((12,len(idx)))
        for i in range(len(idx)):
            (row, col) = idx[i]
            plc_pix[:,i] = plc_vals[:,row,col]
            sw_pix[:,i] = sw[:,row,col]
            if np.nanmax(plc_vals[:,row,col]) >= 88:
            #if np.nanmax(plc_vals[:,row,col]) <= 20:
                print(row, col)
        # dsf pixel
        #row = 219
        #col = 601

        # grw pixel
        row = 188
        col = 666
        get_lat_lon(row, col)

        plc_all = np.append(plc_all, plc_vals[:,row,col])
        sw_all = np.append(sw_all, sw[:,row,col])
        sw_all4 = np.append(sw_all4, sw4[:,row,col])
        lai_all = np.append(lai_all, lai_vals[:,row,col])
        psi_leaf_all = np.append(psi_leaf_all, psi_leaf_vals[:,row,col])
        psi_stem_all = np.append(psi_stem_all, psi_stem_vals[:,row,col])
        weighted_psi_soil_all = np.append(weighted_psi_soil_all, weighted_psi_soil_vals[:,row,col])
        rain_all = np.append(rain_all, Rainf_vals[:,row,col])


    fig, axs = plt.subplots(5, figsize=(8,10))
    print(np.sum(rain_all * 86400 * 30), np.sum(rain_all * 86400 * 30) / 11)
    axs[0].plot(rain_all * 86400 * 30.)
    axs[1].plot(plc_all)
    axs[1].set_ylim(0, 90)
    axs[2].plot(sw_all)
    axs[2].plot(sw_all4, label="top 4 layers")
    axs[2].legend(numpoints=1, loc="best")
    axs[3].plot(psi_leaf_all, label="leaf")
    axs[3].plot(psi_stem_all, label="stem")
    axs[3].legend(numpoints=1, loc="best")
    axs[4].plot(weighted_psi_soil_all)

    plt.show()
    sys.exit()



if __name__ == "__main__":

    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    main(plot_dir)

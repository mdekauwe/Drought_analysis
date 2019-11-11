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


    for year in np.arange(2000, 2007):
    #for year in np.arange(2000, 2001):
    #for year in np.arange(2000, 2002):

        fdir = "/Users/mdekauwe/Desktop/outputs"

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


        plc_rf = np.mean(plc_rf, axis=1)
        plc_wsf = np.mean(plc_wsf, axis=1)
        plc_dsf = np.mean(plc_dsf, axis=1)
        plc_grw = np.mean(plc_grw, axis=1)
        plc_saw = np.mean(plc_saw, axis=1)

        sw_rf = np.mean(sw_rf, axis=1)
        sw_wsf = np.mean(sw_wsf, axis=1)
        sw_dsf = np.mean(sw_dsf, axis=1)
        sw_grw = np.mean(sw_grw, axis=1)
        sw_saw = np.mean(sw_saw, axis=1)

        plc_rf_all = np.append(plc_rf_all, plc_rf)
        plc_wsf_all = np.append(plc_wsf_all, plc_wsf)
        plc_dsf_all = np.append(plc_dsf_all, plc_dsf)
        plc_grw_all = np.append(plc_grw_all, plc_grw)
        plc_saw_all = np.append(plc_saw_all, plc_saw)

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

    plt.scatter(sw_rf_all, plc_rf_all, label="RF")
    plt.scatter(sw_wsf_all, plc_wsf_all, label="WSF")
    plt.scatter(sw_dsf_all, plc_dsf_all, label="DSF")
    plt.scatter(sw_grw_all, plc_grw_all, label="GRW")
    plt.scatter(sw_saw_all, plc_saw_all, label="SAW")

    plt.legend()

    plt.show()
    sys.exit()



if __name__ == "__main__":

    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    main(plot_dir)

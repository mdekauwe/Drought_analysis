#!/usr/bin/env python

"""
Cluster the height of Eucs using K-Means clustering

Data from -> https://landscape.jpl.nasa.gov/

That's all folks.
"""

__author__ = "Martin De Kauwe"
__version__ = "1.0 (23.04.2019)"
__email__ = "mdekauwe@gmail.com"

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from sklearn.cluster import KMeans
import gdal
import xarray as xr

def get_heights(fn):

    src_ds = gdal.Open(fn)
    band = src_ds.GetRasterBand(1)

    cols = src_ds.RasterXSize
    rows = src_ds.RasterYSize
    transform = src_ds.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = -transform[5]
    ulx, xres, xskew, uly, yskew, yres  = src_ds.GetGeoTransform()
    lrx = ulx + (src_ds.RasterXSize * xres)
    lry = uly + (src_ds.RasterYSize * yres)

    lats = np.linspace(uly, lry, rows)
    lons = np.linspace(ulx, lrx, cols)

    lonx, laty = np.meshgrid(lats, lons)

    latx = np.ones((len(lats),len(lons))).shape

    data = band.ReadAsArray(0, 0, cols, rows)

    """
    # gdalwarp -tr 0.00833333 0.00833333 Global_l3c_error_map.tif \
    #                                    Global_l3c_error_map_inter.tif
    fn_error = "Global_l3c_error_map_inter.tif"

    src_ds = gdal.Open(os.path.join(fdir, fn_error))
    band_error = src_ds.GetRasterBand(1)

    cols_error = src_ds.RasterXSize
    rows_error = src_ds.RasterYSize
    transform = src_ds.GetGeoTransform()
    (ulx_error, xres_error,
     xskew_error, uly_error,
     yskew_error, yres_error)  = src_ds.GetGeoTransform()
    lrx_error = ulx_error + (src_ds.RasterXSize * xres_error)
    lry_error = uly_error + (src_ds.RasterYSize * yres_error)
    lats_error = np.linspace(lry_error, uly_error, rows_error)
    data_error = band_error.ReadAsArray(0, 0, cols_error, rows_error)

    # Screen by error dataset, set to zero as we mask this below
    data = np.where(data_error < 0.0, 0.0, data)
    """

    idy = np.argwhere((lats>=-43.6345972634) & (lats<-10.6681857235))
    idx = np.argwhere((lons>=113.338953078) & (lons<153.569469029))

    aus = data[idy.min():idy.max(),idx.min():idx.max()]
    aus_lat = lats[idy.min():idy.max()]
    aus_lon = lons[idx.min():idx.max()]

    #plt.imshow(np.flipud(aus))
    #plt.colorbar()
    #plt.show()
    #sys.exit()

    return (aus, aus_lat, aus_lon)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

if __name__ == "__main__":

    fn_cab = "/Users/mdekauwe/Desktop/drought_desktop/outputs/cable_out_2000.nc"
    ds = xr.open_dataset(fn_cab)
    iveg = ds["iveg"][:,:].values
    idx_rf = np.argwhere(iveg == 18.0)
    idx_wsf = np.argwhere(iveg == 19.0)
    idx_dsf = np.argwhere(iveg == 20.0)
    idx_grw = np.argwhere(iveg == 21.0)
    idx_saw = np.argwhere(iveg == 22.0)

    #gdalwarp -tr 0.05 0.05 Simard_Pinto_3DGlobalVeg_JGR.tif \
    #                       Simard_Pinto_3DGlobalVeg_JGR_5km.tif
    fn = "Simard_Pinto_3DGlobalVeg_JGR_5km.tif"

    (aus, aus_lat, aus_lon) = get_heights(fn)

    rf_heights = []
    for i in range(len(idx_rf)):
        (row, col) = idx_rf[i]

        lat = ds["latitude"][row,col].values
        lon = ds["longitude"][row,col].values

        r = find_nearest(aus_lat, lat)
        c = find_nearest(aus_lon, lon)
        height = aus[r,c]
        if height > 0.0:
            rf_heights.append(height)
    rf_heights = np.array(rf_heights)
    print("RF: ", np.mean(rf_heights), np.median(rf_heights))
    #plt.hist(rf_heights)
    #plt.show()

    wsf_heights = []
    for i in range(len(idx_wsf)):
        (row, col) = idx_wsf[i]

        lat = ds["latitude"][row,col].values
        lon = ds["longitude"][row,col].values

        r = find_nearest(aus_lat, lat)
        c = find_nearest(aus_lon, lon)
        height = aus[r,c]
        if height > 0.0:
            wsf_heights.append(height)
    wsf_heights = np.array(wsf_heights)
    print("WSF: ", np.mean(wsf_heights), np.median(wsf_heights))

    dsf_heights = []
    for i in range(len(idx_dsf)):
        (row, col) = idx_dsf[i]

        lat = ds["latitude"][row,col].values
        lon = ds["longitude"][row,col].values

        r = find_nearest(aus_lat, lat)
        c = find_nearest(aus_lon, lon)
        height = aus[r,c]
        if height > 0.0:
            dsf_heights.append(height)
    dsf_heights = np.array(dsf_heights)
    print("DSF: ", np.mean(dsf_heights), np.median(dsf_heights))

    grw_heights = []
    for i in range(len(idx_grw)):
        (row, col) = idx_grw[i]

        lat = ds["latitude"][row,col].values
        lon = ds["longitude"][row,col].values

        r = find_nearest(aus_lat, lat)
        c = find_nearest(aus_lon, lon)
        height = aus[r,c]
        if height > 0.0:
            grw_heights.append(height)
    grw_heights = np.array(grw_heights)
    print("GRW: ", np.mean(grw_heights), np.median(grw_heights))

    saw_heights = []
    for i in range(len(idx_saw)):
        (row, col) = idx_saw[i]

        lat = ds["latitude"][row,col].values
        lon = ds["longitude"][row,col].values

        r = find_nearest(aus_lat, lat)
        c = find_nearest(aus_lon, lon)
        height = aus[r,c]
        if height > 0.0:
            saw_heights.append(height)
    saw_heights = np.array(saw_heights)
    print("SAW: ", np.mean(saw_heights), np.median(saw_heights))

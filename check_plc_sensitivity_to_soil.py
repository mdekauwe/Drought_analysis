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

def main(fname1, fname2, plot_dir):

    ds1 = xr.open_dataset(fname1)
    lat = ds1.y.values
    lon = ds1.x.values
    bottom, top = lat[0], lat[-1]
    left, right = lon[0], lon[-1]

    plc_csiro = ds1.plc[:,0,:,:].values
    plc_csiro = np.nanmax(plc_csiro, axis=0)


    ds2 = xr.open_dataset(fname2)
    plc_open = ds2.plc[:,0,:,:].values
    plc_open = np.nanmax(plc_open, axis=0)

    fig = plt.figure(figsize=(9, 6))
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.size'] = "14"
    plt.rcParams['font.sans-serif'] = "Helvetica"

    cmap = plt.cm.get_cmap('YlOrRd', 9) # discrete colour map
    #cmap = plt.cm.YlOrRd

    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes, dict(map_projection=projection))
    rows = 1
    cols = 1

    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(rows, cols),
                    axes_pad=0.2,
                    cbar_location='right',
                    cbar_mode='single',
                    cbar_pad=0.5,
                    cbar_size='5%',
                    label_mode='')  # note the empty label_mode


    for i, ax in enumerate(axgr):
        # add a subplot into the array of plots
        #ax = fig.add_subplot(rows, cols, i+1, projection=ccrs.PlateCarree())
        plims = plot_map(ax, plc_csiro-plc_open, cmap, i, top, bottom, left, right)
        #plims = plot_map(ax, ds.plc[0,0,:,:], cmap, i)


    cbar = axgr.cbar_axes[0].colorbar(plims)
    cbar.ax.set_title("%", fontsize=16)

    ofname = os.path.join(plot_dir, "plc_diff_due_to_soil.png")

    fig.savefig(ofname, dpi=300, bbox_inches='tight',
                pad_inches=0.1)
    #plt.show()

def plot_map(ax, var, cmap, i, top, bottom, left, right):
    vmin, vmax = -10, 10
    #top, bottom = 90, -90
    #left, right = -180, 180
    img = ax.imshow(var, origin='lower',
                    transform=ccrs.PlateCarree(),
                    interpolation='nearest', cmap=cmap,
                    extent=(left, right, bottom, top),
                    vmin=vmin, vmax=vmax)
    ax.coastlines(resolution='10m', linewidth=1.0, color='black')
    #ax.add_feature(cartopy.feature.OCEAN)

    ax.set_xlim(140, 154)
    ax.set_ylim(-39.4, -28)

    if i == 0 or i >= 5:

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='black', alpha=0.5,
                          linestyle='--')
    else:
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                          linewidth=0.5, color='black', alpha=0.5,
                          linestyle='--')

    #if i < 5:
    #s    gl.xlabels_bottom = False
    if i > 5:
        gl.ylabels_left = False

    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlines = False
    gl.ylines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    gl.xlocator = mticker.FixedLocator([141, 145,  149, 153])
    gl.ylocator = mticker.FixedLocator([-29, -32, -35, -38])

    return img


if __name__ == "__main__":

    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)


    fname1 = "outputs/all_yrs_plc.nc"
    fname2 = "../AWAP_SE_aus_hydraulics_ebf_copernicus/outputs/all_yrs_plc.nc"
    main(fname1, fname2, plot_dir)

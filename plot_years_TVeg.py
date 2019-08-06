#!/usr/bin/env python
"""
Plot Jan to Dec TVeg
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

def main(fname, plot_dir):

    ds = xr.open_dataset(fname)

    fig = plt.figure(figsize=(20, 8))
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.size'] = "14"
    plt.rcParams['font.sans-serif'] = "Helvetica"

    cmap = plt.cm.viridis
    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes,
                  dict(map_projection=projection))
    rows = 2
    cols = 6

    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(rows, cols),
                    axes_pad=0.6,
                    cbar_location='right',
                    cbar_mode='single',
                    cbar_pad=0.5,
                    cbar_size='5%',
                    label_mode='')  # note the empty label_mode

    month = 1
    for i, ax in enumerate(axgr):
        # add a subplot into the array of plots
        #ax = fig.add_subplot(rows, cols, i+1, projection=ccrs.PlateCarree())
        plims = plot_map(ax, ds.TVeg[i,:,:,], month, cmap, i)

        month += 1

    cbar = axgr.cbar_axes[0].colorbar(plims)
    cbar.ax.set_title("Transpiration\n(mm d$^{-1}$)", fontsize=16)

    ofname = os.path.join(plot_dir, "Months_transpiration.png")
    fig.savefig(ofname, dpi=150, bbox_inches='tight',
                pad_inches=0.1)

def plot_map(ax, var, month, cmap, i):
    vmin, vmax = 0.0, 3.0
    top, bottom = 90, -90
    left, right = -180, 180
    img = ax.imshow(var, origin='lower',
                    transform=ccrs.PlateCarree(),
                    interpolation='nearest', cmap=cmap,
                    extent=(left, right, bottom, top),
                    vmin=vmin, vmax=vmax)
    ax.coastlines(resolution='10m', linewidth=1.0, color='black')
    #ax.add_feature(cartopy.feature.OCEAN)
    ax.set_title("%d" % (month), fontsize=16)
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

    if i < 6:
        gl.xlabels_bottom = False
    if i > 6:
        gl.ylabels_left = False

    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlines = False
    gl.ylines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    gl.xlocator = mticker.FixedLocator([141, 145,  149, 153])
    gl.ylocator = mticker.FixedLocator([-29, -32, -35, -38])

    if i == 0 :
        ax.text(-0.25, -0.25, 'Latitude', va='bottom', ha='center',
                rotation='vertical', rotation_mode='anchor',
                transform=ax.transAxes, fontsize=16)
    if i == 9:
        ax.text(0.5, -0.25, 'Longitude', va='bottom', ha='center',
                rotation='horizontal', rotation_mode='anchor',
                transform=ax.transAxes, fontsize=16)

    return img


if __name__ == "__main__":

    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    fname = "outputs/cable_out_1995.nc"

    main(fname, plot_dir)

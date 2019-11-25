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

def main(fname, plot_dir):

    ds = xr.open_dataset(fname)
    lat = ds.y.values
    lon = ds.x.values
    bottom, top = lat[0], lat[-1]
    left, right = lon[0], lon[-1]

    #plt.imshow(ds.LAI[0,:,:,])
    #plt.colorbar()
    #plt.show()
    #sys.exit()
    fig = plt.figure(figsize=(20, 8))
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.size'] = "14"
    plt.rcParams['font.sans-serif'] = "Helvetica"

    #cmap = plt.cm.viridis
    cmap = plt.cm.Greens
    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes,
                  dict(map_projection=projection))
    rows = 1
    cols = 1

    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(rows, cols),
                    axes_pad=0.6,
                    cbar_location='right',
                    cbar_mode='single',
                    cbar_pad=0.5,
                    cbar_size='5%',
                    label_mode='')  # note the empty label_mode

    year = 2000
    for i, ax in enumerate(axgr):
        # add a subplot into the array of plots
        #ax = fig.add_subplot(rows, cols, i+1, projection=ccrs.PlateCarree())
        plims = plot_map(ax, ds.LAI[i,:,:,], year, cmap, i, top, bottom,
                         left, right)

        year += 1

    cbar = axgr.cbar_axes[0].colorbar(plims)
    cbar.ax.set_title("LAI\n(m$^{2}$ m$^{-2}$)", fontsize=16)

    ofname = os.path.join(plot_dir, "DJF_lai.png")
    fig.savefig(ofname, dpi=150, bbox_inches='tight',
                pad_inches=0.1)

def plot_map(ax, var, year, cmap, i, top, bottom, left, right):
    vmin, vmax = 0.0, 5.0
    print(np.nanmax(var))
    img = ax.imshow(var, origin='lower',
                    transform=ccrs.PlateCarree(),
                    interpolation='nearest', cmap=cmap,
                    extent=(left, right, bottom, top),
                    vmin=vmin, vmax=vmax)
    ax.coastlines(resolution='10m', linewidth=1.0, color='black')
    #ax.add_feature(cartopy.feature.OCEAN)
    #ax.set_title("%d-%d" % (year, year+1), fontsize=16)
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

    if i == 0 :
        ax.text(-0.1, 0.5, 'Latitude', va='bottom', ha='center',
                rotation='vertical', rotation_mode='anchor',
                transform=ax.transAxes, fontsize=16)
    if i == 0:
        ax.text(0.5, -0.1, 'Longitude', va='bottom', ha='center',
                rotation='horizontal', rotation_mode='anchor',
                transform=ax.transAxes, fontsize=16)

    return img


if __name__ == "__main__":

    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    fname = "outputs/djf_lai.nc"
    #fname = "../djf.nc"

    main(fname, plot_dir)

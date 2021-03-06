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
import cartopy.io.shapereader as shpreader

def main(fname, lai_fname, plot_dir, plc_type=None):

    ds = xr.open_dataset(fname)
    lat = ds.y.values
    lon = ds.x.values
    bottom, top = lat[0], lat[-1]
    left, right = lon[0], lon[-1]

    plc = ds.plc[:,0,:,:].values

    #dsx = xr.open_dataset(lai_fname)
    #lai = dsx["LAI"][:,:,:].values
    #lai = np.max(lai, axis=0)

    #plc = np.nanmean(plc, axis=0)
    #plt.imshow(plc[:,:])
    #plt.imshow(plc[10,:,:])
    #plt.colorbar()
    #plt.show()
    #sys.exit()
    if plc_type == "mean":
        plc = np.nanmean(plc, axis=0)
    elif plc_type == "max":
        plc = np.nanmax(plc, axis=0)
    elif plc_type == "median":
        plc = np.nanmedian(plc, axis=0)

    #plc = np.where(lai<0.05, np.nan, plc)

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
        plims = plot_map(ax, plc, cmap, i, top, bottom, left, right)
        #plims = plot_map(ax, ds.plc[0,0,:,:], cmap, i)

        #import cartopy.feature as cfeature
        #states = cfeature.NaturalEarthFeature(category='cultural',
        #                                      name='.in_1_states_provinces_lines',
        #                                      scale='10m',facecolor='none')

        # plot state border
        #SOURCE = 'Natural Earth'
        #LICENSE = 'public domain'
        #ax.add_feature(states, edgecolor='black', lw=0.5)

        from cartopy.feature import ShapelyFeature
        from cartopy.io.shapereader import Reader
        #fname = '/Users/mdekauwe/research/Drought_linkage/Bios2_SWC_1979_2013/AUS_shape/STE11aAust.shp'
        fname = "/Users/mdekauwe/Dropbox/ne_10m_admin_1_states_provinces_lines/ne_10m_admin_1_states_provinces_lines.shp"
        shape_feature = ShapelyFeature(Reader(fname).geometries(),
                                       ccrs.PlateCarree(), edgecolor='black')
        ax.add_feature(shape_feature, facecolor='none', edgecolor='black',
                       lw=0.5)

    cbar = axgr.cbar_axes[0].colorbar(plims)
    cbar.ax.set_title("PLC (%)", fontsize=16, pad=10)

    if plc_type == "mean":
        ofname = os.path.join(plot_dir, "plc_mean.png")
    elif plc_type == "max":
        ofname = os.path.join(plot_dir, "plc_max.png")
    elif plc_type == "median":
        ofname = os.path.join(plot_dir, "plc_median.png")


    props = dict(boxstyle='round', facecolor='white', alpha=0.0, ec="white")
    ax.text(0.95, 0.05, "(a)", transform=ax.transAxes, fontsize=12,
             verticalalignment='top', bbox=props)


    fig.savefig(ofname, dpi=300, bbox_inches='tight',
                pad_inches=0.1)
    #plt.show()

def plot_map(ax, var, cmap, i, top, bottom, left, right):
    vmin, vmax = 0, 90 #88
    #top, bottom = 90, -90
    #left, right = -180, 180
    img = ax.imshow(var, origin='lower',
                    transform=ccrs.PlateCarree(),
                    interpolation=None, cmap=cmap,
                    extent=(left, right, bottom, top),
                    vmin=vmin, vmax=vmax)
    ax.coastlines(resolution='10m', linewidth=1.0, color='black')
    #ax.add_feature(cartopy.feature.OCEAN)

    ax.set_xlim(140.7, 154)
    ax.set_ylim(-39.2, -28.1)

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


    #fname = "outputs/min_plc.nc"
    fname = "outputs/all_yrs_plc.nc"
    lai_fname = "outputs/cable_out_2000.nc"
    main(fname, lai_fname, plot_dir, plc_type="mean")
    main(fname, lai_fname, plot_dir, plc_type="max")
    #main(fname, lai_fname, plot_dir, plc_type="median")

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

def main(fname, plot_dir, start_year, end_year):

    ds = xr.open_dataset(fname)
    lat = ds.y.values
    lon = ds.x.values
    bottom, top = lat[0], lat[-1]
    left, right = lon[0], lon[-1]

    nmonths, nrows, ncols = ds.Rainf.shape
    nyears = (end_year - start_year) + 1
    yr_count = 0
    aet = np.zeros((nyears,nrows,ncols))
    ppt = np.zeros((nyears,nrows,ncols))
    sec_2_day = 86400.0
    count = 0
    yr_count = 0
    mth_count = 1

    for year in np.arange(start_year, end_year+1):
        for month in np.arange(1, 13):

            days_in_month = monthrange(year, month)[1]
            conv = sec_2_day * days_in_month

            yr_val = str(ds.time[count].values).split("-")[0]
            print(yr_val)

            aet[yr_count,:,:] += ds.Evap[count,:,:] * conv
            ppt[yr_count,:,:] += ds.Rainf[count,:,:] * conv
            mth_count += 1

            if mth_count == 13:
                mth_count = 1
                yr_count += 1

            count += 1

    ppt = np.nanmean(ppt, axis=0)
    aet = np.nanmean(aet, axis=0)
    cmi = np.where(~np.isnan(aet), ppt-aet, np.nan)

    fig = plt.figure(figsize=(9, 6))
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.size'] = "14"
    plt.rcParams['font.sans-serif'] = "Helvetica"

    cmap = plt.cm.get_cmap('BrBG', 10) # discrete colour map

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
        plims = plot_map(ax, cmi, cmap, i, top, bottom, left, right)

        """
        import cartopy.feature as cfeature
        states = cfeature.NaturalEarthFeature(category='cultural',
                                              name='.in_1_states_provinces_lines',
                                              scale='10m',facecolor='none')

        # plot state border
        SOURCE = 'Natural Earth'
        LICENSE = 'public domain'
        ax.add_feature(states, edgecolor='black', lw=0.5)
        """
        from cartopy.feature import ShapelyFeature
        from cartopy.io.shapereader import Reader
        #fname = '/Users/mdekauwe/research/Drought_linkage/Bios2_SWC_1979_2013/AUS_shape/STE11aAust.shp'
        fname = "/Users/mdekauwe/Dropbox/ne_10m_admin_1_states_provinces_lines/ne_10m_admin_1_states_provinces_lines.shp"
        shape_feature = ShapelyFeature(Reader(fname).geometries(),
                                       ccrs.PlateCarree(), edgecolor='black')
        ax.add_feature(shape_feature, facecolor='none', edgecolor='black',
                       lw=0.5)

    cbar = axgr.cbar_axes[0].colorbar(plims)
    cbar.ax.set_title("P-AET\n(mm yr$^{-1}$)", fontsize=16, pad=10)
    cbar.ax.set_yticklabels([' ', '$\minus$40', '$\minus$20', '0', '20', '40-1200'])

    props = dict(boxstyle='round', facecolor='white', alpha=0.0, ec="white")
    ax.text(0.95, 0.05, "(c)", transform=ax.transAxes, fontsize=12,
             verticalalignment='top', bbox=props)

    ofname = os.path.join(plot_dir, "cmi.png")
    fig.savefig(ofname, dpi=300, bbox_inches='tight',
                pad_inches=0.1)

def plot_map(ax, var, cmap, i, top, bottom, left, right):
    print(np.nanmin(var), np.nanmax(var))
    vmin, vmax = -50, 50


    #top, bottom = 90, -90
    #left, right = -180, 180
    img = ax.imshow(var, origin='lower',
                    transform=ccrs.PlateCarree(),
                    interpolation='nearest', cmap=cmap,
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

    fname = "outputs/all_yrs_CMI.nc"

    main(fname, plot_dir, 2000, 2009)

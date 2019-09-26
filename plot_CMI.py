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

def main(fname, plot_dir):

    ds = xr.open_dataset(fname)

    """
    nmonths, nrows, ncols = ds.Rainf.shape

    cmi = np.zeros((nmonths, nrows, ncols))
    cnt = 0
    sec_2_day = 86400.0
    for year in np.arange(2000, 2010):
        for month in np.arange(1, 13):
            days_in_month = monthrange(year, month)[1]
            conv = sec_2_day * days_in_month
            cmi[cnt,:,:] = (ds.Rainf[cnt,:,:] * conv) - \
                            (ds.Evap[cnt,:,:] * conv)


            cnt = cnt + 1

    cmi = np.sum(cmi, axis=0)
    """

    nmonths, nrows, ncols = ds.Rainf.shape
    nyears = 10
    pet = np.zeros((nyears,nrows,ncols))
    ppt = np.zeros((nyears,nrows,ncols))
    sec_2_day = 86400.0
    count = 0.0
    for year in np.arange(2000, 2010):
        #print(year)
        for month in np.arange(1, 13):

            days_in_month = monthrange(year, month)[1]
            conv = sec_2_day * days_in_month

            if year == 2000 and month >= 7:

                pet[yr_count,:,:] += ds.PET[count,:,:] * conv
                ppt[yr_count,:,:] += ds.Rainf[count,:,:] * conv



                mth_count += 1

            elif year > 2000 and year <= 2009:

                pet[yr_count,:,:] += ds.PET[count,:,:] * conv
                ppt[yr_count,:,:] += ds.Rainf[count,:,:] * conv
                mth_count += 1

            elif year == 2009 and month <= 6:

                pet[yr_count,:,:] += ds.PET[count,:,:] * conv
                ppt[yr_count,:,:] += ds.Rainf[count,:,:] * conv
                mth_count += 1



            if mth_count == 13:
                mth_count = 1
                yr_count += 1


            count += 1

    ppt = np.mean(ds_ppt.precip[0:count,:,:], axis=0)
    pet = np.mean(ds_pet.PET[0:count,:,:], axis=0)
    cmi = ppt - pet

    # just keep deficit areas
    #cmi = np.where(cmi >= 300., np.nan, cmi)

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
        #ax = fig.add_subplot(rows, cols, i+1, projection=ccrs.PlateCarree())
        plims = plot_map(ax, cmi / 10, cmap, i)
        #plims = plot_map(ax, ds.plc[0,0,:,:], cmap, i)


    cbar = axgr.cbar_axes[0].colorbar(plims)
    cbar.ax.set_title("P-AET\n(mm y$^{-1}$)", fontsize=16)

    ofname = os.path.join(plot_dir, "cmi.png")
    fig.savefig(ofname, dpi=150, bbox_inches='tight',
                pad_inches=0.1)

def plot_map(ax, var, cmap, i):
    print(np.nanmin(var), np.nanmax(var))
    vmin, vmax = -100, 100
    top, bottom = 90, -90
    left, right = -180, 180
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

    fname = "outputs/all_yrs_CMI.nc"

    main(fname, plot_dir)

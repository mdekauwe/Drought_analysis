#!/usr/bin/env python
"""
Plot DJF timeseries comparison for the Millennium drought
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (25.07.2019)"
__email__ = "mdekauwe@gmail.com"

import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import sys

def main(fname, plot_dir):

    ds = xr.open_dataset(fname)
    print(ds)

    time = ds.time
    TVeg = ds.TVeg[:,100:127,634:670].values * 86400
    LAI = ds.LAI[:,100:127,634:670].values
    iveg = ds.iveg[100:127,634:670].values
    gpp = ds.GPP[:,100:127,634:670].values
    qle = ds.Qle[:,100:127,634:670].values
    Rainf = ds.Rainf[:,100:127,634:670].values * 86400
    Tair = ds.Tair[:,100:127,634:670].values - 273.15
    froot = ds.froot[:,100:127,634:670].values
    #plt.imshow(TVeg[1,:,:])
    #plt.imshow(LAI[1,:,:])
    #plt.imshow(iveg[:,:])
    #plt.imshow(gpp[0,:,:])
    #plt.imshow(qle[2,:,:])
    #plt.imshow(Rainf[1,:,:])
    plt.imshow(froot[2,:,:])
    plt.colorbar()
    plt.show()
    sys.exit()

    width = 9
    height = 6
    fig = plt.figure(figsize=(width, height))
    fig.subplots_adjust(hspace=0.05)
    fig.subplots_adjust(wspace=0.02)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['font.size'] = 16
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16

    ax = fig.add_subplot(111)


    ax.plot(time, hyd*86400, ls="-", label="Hydraulics")
    ax.plot(time, ctl*86400, ls="-", label="Control")
    ax.legend(numpoints=1, ncol=1, frameon=False, loc="best")
    ax.set_ylabel('E (mm d$^{-1}$)')

    ax.set_ylim(0, 3.5)
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_major_locator(MaxNLocator(10))

    ofname = os.path.join(plot_dir, "Transpiration_test_timeseries.png")
    fig.savefig(ofname, dpi=150, bbox_inches='tight',
                pad_inches=0.1)

if __name__ == "__main__":

    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    fname = "outputs/cable_out_1995.nc"

    main(fname, plot_dir)

#!/usr/bin/env python
"""
Input forcing for CABLE is 3 hrly, linearly interpolate it to 30 mins.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (11.09.2019)"
__email__ = "mdekauwe@gmail.com"

import os
import xarray as xr
import numpy as np
import sys
import glob

def interpolate_forcing(fpath, var, start_date, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = glob.glob(os.path.join(fpath, "%s/*.nc") % (var))

    #for
    #print(files)

    #ds = xr.open_dataset(fname)
    #ds = ds.interp(time=48, method="linear")
    #ds = ds.resample(time='30min').interpolate('linear')
    #print(ds)

    #
    #cdo seldate,1995-12-31 single_pixel_gswp3_forcing.nc tmp.nc
    #cdo settaxis,1996-01-01,00:00:00,3hours tmp.nc tmp2.nc
    #cdo mergetime single_pixel_gswp3_forcing.nc tmp2.nc tmp3.nc

    #cdo inttime,1995-01-01,00:00:00,30minutes tmp3.nc tmp4.nc
    #cdo seldate,1995-01-01,1995-12-31 tmp4.nc single_pixel_gswp3_forcing_int.nc

    #rm tmp*.nc


    """
    cmd = './%s %s > /dev/null 2>&1' % (self.cable_exe, nml_fname)
    error = subprocess.call(cmd, shell=True)
    if error is 1:
        print("Job failed to submit")
    """

if __name__ == "__main__":

    #fpath = "/g/data1/wd9/MetForcing/Global/GSWP3_2017/"
    #vars = ["LWdown", "PSurf", "SWdown", "Tair", "Qair", \
    #         "Rainf", "Snowf", "Wind"]
    fpath = "/Users/mdekauwe/Desktop/GSWP3"
    output_dir = "/Users/mdekauwe/Desktop/test"
    vars = ["Tair"]
    start_date = "1995-01-01,00:00:00"
    for var in vars:
        interpolate_forcing(fpath, var, start_date, output_dir)

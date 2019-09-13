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
import subprocess

def interpolate_forcing(fpath, var, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_dir_var = os.path.join(output_dir, var)
    if not os.path.exists(output_dir_var):
        os.makedirs(output_dir_var)

    files = glob.glob(os.path.join(fpath, "%s/*.nc") % (var))
    years = np.sort(np.asarray([int(f[-7:-3]) for f in files]))

    last_year = years[-1]
    start_date = "%d-01-01,00:00:00" % (years[0])

    for year in years:

        fn = os.path.join(fpath,
                          "%s/GSWP3.BC.%s.3hrMap.%d.nc" % (var, var, year))

        if year != last_year:
            print(fn)

            nxt_fn = os.path.join(output_dir_var,
                                  "GSWP3.BC.%s.3hrMap.%d.nc" % (var, year + 1))
            start_date = "%d-01-01,00:00:00" % (year)

            cmd = "cdo inttime,%s,30minutes %s %s" % (start_date, fn, nxt_fn)
            error = subprocess.call(cmd, shell=True)
            if error is 1:
                raise Exception("Error interpolating file")
        else:

            # Get the last day
            last_date = "%d-12-31,00:00:00" % (year)
            tmp_fn = os.path.join(output_dir_var, "tmp.nc")
            cmd = "cdo seldate,%s %s %s" % (last_date, fn, tmp_fn)
            error = subprocess.call(cmd, shell=True)
            if error is 1:
                raise Exception("Error getting the last day")

            # change the date
            tmp2_fn = os.path.join(output_dir_var, "tmp2.nc")
            new_date = "%d-01-01,00:00:00" % (year + 1)
            cmd = "cdo settaxis,%s,3hours %s %s" % (new_date, tmp_fn, tmp2_fn)
            error = subprocess.call(cmd, shell=True)
            if error is 1:
                raise Exception("Error changing the date file")

            # merge dummy date
            tmp3_fn = os.path.join(output_dir_var, "tmp3.nc")
            cmd = "cdo mergetime %s %s %s" (fn, tmp2_fn, tmp3_fn)
            error = subprocess.call(cmd, shell=True)
            if error is 1:
                raise Exception("Error merging new file")

            nxt_fn = os.path.join(output_dir_var,
                                  "GSWP3.BC.%s.3hrMap.%d.nc" % (var, year + 1))
            start_date = "%d-01-01,00:00:00" % (year)

            cmd = "cdo inttime,%s,30minutes %s %s" % (start_date, fn, nxt_fn)
            error = subprocess.call(cmd, shell=True)
            if error is 1:
                raise Exception("Error interpolating file")
                
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

    for var in vars:
        interpolate_forcing(fpath, var, output_dir)

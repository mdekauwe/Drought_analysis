#!/usr/bin/env python

"""
Reset the PLC after spinup so that we start from 0.

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (26.09.2019)"
__email__ = "mdekauwe@gmail.com"

import pandas as pd
import numpy as np
import xarray as xr
import os
import sys
import shutil

def main(restart_dir, year):

    fn = os.path.join(restart_dir, "restart_%d.nc" % (year))
    ds = xr.open_dataset(fn)

    ds['plc'][:] = 0.0

    ofname = "temp.nc"
    ds.to_netcdf(ofname)
    ds.close()
    shutil.move(ofname, fn)

if __name__ == "__main__":

    restart_dir = "restarts"
    year = 1999
    main(restart_dir, year)

#!/bin/bash

cdo mergetime outputs/cable_out_*.nc outputs/all_yrs.nc

cdo select,name=iveg outputs/cable_out_2000.nc outputs/tmp.nc

# Fix the longitude issue in the CABLE output files...
cdo sellonlatbox,-180,180,-90,90 outputs/tmp.nc outputs/iveg.nc

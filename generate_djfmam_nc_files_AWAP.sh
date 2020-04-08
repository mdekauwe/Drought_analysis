#!/bin/bash

cdo mergetime outputs/cable_out_*.nc outputs/all_yrs.nc

cdo select,name=TVeg,LAI,GPP outputs/all_yrs.nc outputs/tmp_all_yrs.nc
mv outputs/tmp_all_yrs.nc outputs/all_yrs.nc

# Fix the longitude issue in the CABLE output files...
cdo sellonlatbox,112,154,-44,-10 outputs/all_yrs.nc outputs/all_yrs_fixed_long.nc
mv outputs/all_yrs_fixed_long.nc outputs/all_yrs.nc

#6 mean over 6 months (December to February & March to May)
#11 skip the first 11 months (January to November)
#6 skip 6 months between every 6 months interval (June to November)
cdo mulc,86400 -timselmean,6,11,6 outputs/all_yrs.nc outputs/djfmam.nc

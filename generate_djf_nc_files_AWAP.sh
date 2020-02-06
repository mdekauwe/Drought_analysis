#!/bin/bash

cdo mergetime outputs/cable_out_*.nc outputs/all_yrs.nc

cdo select,name=TVeg,LAI,GPP outputs/all_yrs.nc outputs/tmp_all_yrs.nc
mv outputs/tmp_all_yrs.nc outputs/all_yrs.nc

# Fix the longitude issue in the CABLE output files...
cdo sellonlatbox,112,154,-44,-10 outputs/all_yrs.nc outputs/all_yrs_fixed_long.nc
mv outputs/all_yrs_fixed_long.nc outputs/all_yrs.nc

#mm/s to mm/d
#3 mean over 3 months (December to February)
#11 skip the first 11 months (January to November)
#9 skip 9 months between every 3 months interval (March to November)
cdo mulc,86400 -timselmean,3,11,9 outputs/all_yrs.nc outputs/djf.nc


cdo mergetime outputs/cable_out_*.nc outputs/all_yrs.nc
cdo select,name=TVeg,LAI,GPP,Fwsoil outputs/all_yrs.nc outputs/tmp_all_yrs.nc
mv outputs/tmp_all_yrs.nc outputs/all_yrs.nc

# Fix the longitude issue in the CABLE output files...
cdo sellonlatbox,112,154,-44,-10 outputs/all_yrs.nc outputs/all_yrs_fixed_long.nc
mv outputs/all_yrs_fixed_long.nc outputs/all_yrs.nc

cdo timselmean,3,11,9 outputs/all_yrs.nc outputs/djf_lai.nc

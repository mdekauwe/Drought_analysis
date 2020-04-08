#!/bin/bash

cdo mergetime outputs/cable_out_*.nc outputs/all_yrs_plc.nc

cdo select,name=plc outputs/all_yrs_plc.nc outputs/tmp_all_yrs.nc
mv outputs/tmp_all_yrs.nc outputs/all_yrs_plc.nc

# Fix the longitude issue in the CABLE output files...
cdo sellonlatbox,-180,180,-90,90 outputs/all_yrs_plc.nc outputs/all_yrs_fixed_long.nc
mv outputs/all_yrs_fixed_long.nc outputs/all_yrs_plc.nc

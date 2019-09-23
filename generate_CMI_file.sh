#!/bin/bash

cdo mergetime outputs/cable_out_*.nc outputs/all_yrs_CMI.nc

cdo select,name=Rainf,Evap outputs/all_yrs_CMI.nc outputs/tmp_all_yrs.nc
mv outputs/tmp_all_yrs.nc outputs/all_yrs_CMI.nc

# Fix the longitude issue in the CABLE output files...
cdo sellonlatbox,-180,180,-90,90 outputs/all_yrs_CMI.nc outputs/all_yrs_fixed_long.nc
mv outputs/all_yrs_fixed_long.nc outputs/all_yrs_CMI.nc

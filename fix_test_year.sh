#!/bin/bash

FN=cable_out_1995.nc
# Fix the longitude issue in the CABLE output files...
cdo sellonlatbox,-180,180,-90,90 outputs/$FN outputs/tmp.nc
mv outputs/tmp.nc outputs/$FN

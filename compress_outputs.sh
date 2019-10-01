#!/bin/bash

conda deactivate sci
module use /g/data3/hh5/public/modules
module load conda

outputs="outputs"
backup_outputs="backup_outputs"
if [ ! -d $backup_outputs ]
then
    mkdir -p $backup_outputs
fi

mv $outputs/cable_out_* $backup_outputs
nccompress -r -o -p -b 500 $backup_outputs

conda deactivate
conda activate sci

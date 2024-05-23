#!/bin/tcsh
#SBATCH --partition=GPU
#SBATCH --nodelist=worker093
#SBATCH --time=00:10:00
#SBATCH --nodes=1

set LogFile=output.log
env > $LogFile
groups >> $LogFile
./custom_mapping >> $LogFile

#!/bin/tcsh
#SBATCH --partition=GPU
#SBATCH --nodelist=worker093
#SBATCH --time=00:05:00
#SBATCH --nodes=1

set LogFile=output.log
env > $LogFile
groups >> $LogFile
./time_data_transfer >> $LogFile

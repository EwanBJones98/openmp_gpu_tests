#!/bin/tcsh
#SBATCH --time=7:00:00
#SBATCH --ntasks=16

set LogFile=output.log
env > $LogFile
groups >> $LogFile
./bin/calculate_pressure >> $LogFile

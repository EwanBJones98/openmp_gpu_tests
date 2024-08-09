#!/bin/tcsh
#SBATCH --partition=GPU
#SBATCH --nodelist=worker093
#SBATCH --time=00:10:00
#SBATCH --nodes=1

set LogFile=output.log
env > $LogFile
groups >> $LogFile
./bin/calculate_pressure 2 5000 100 100 timings/optimised_worksharing.txt >> $LogFile

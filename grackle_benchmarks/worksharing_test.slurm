#!/bin/tcsh
#SBATCH --partition=GPU
#SBATCH --nodelist=worker093
#SBATCH --time=48:00:00
#SBATCH --nodes=1

set LogFile=output.log
env > $LogFile
groups >> $LogFile
#./bin/calculate_pressure 2 100 1000 10 1 100 500 timings/worksharing_test.txt >> $LogFile
./bin/calculate_pressure 2 100 10000 20 10 1000 20 timings/worksharing_test.txt >> $LogFile

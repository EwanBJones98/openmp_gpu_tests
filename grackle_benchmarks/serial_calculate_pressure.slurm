#!/bin/tcsh
#SBATCH --time=7:00:00

set LogFile=output.log
env > $LogFile
groups >> $LogFile
./bin/calculate_pressure 2 1000 100 100 timings/gpu_cpu_serial.txt >> $LogFile

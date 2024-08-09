#!/bin/tcsh
#SBATCH --time=7:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=GPU
#SBATCH --nodelist=worker066

setenv OMP_NUM_THREADS 4

set LogFile=output.log
env > $LogFile
groups >> $LogFile
./bin/calculate_pressure 2 100 100 100 timings/gpu_cpu_serial.txt >> $LogFile

#!/bin/tcsh
#SBATCH --partition=GPU
#SBATCH --nodelist=worker093
#SBATCH --time=7:00:00
#SBATCH --nodes=1

set LogFile=output.log
env > $LogFile
groups >> $LogFile
./matrix_mult >> $LogFile

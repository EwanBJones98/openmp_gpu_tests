#!/bin/tcsh

#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --time=00:01:00
#SBATCH --ntasks=1

set LogFile=output_map_struct.log
env > $LogFile
groups >> $LogFile

./map_struct >> $LogFile
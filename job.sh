#!/bin/bash
#PBS -N poisson 
#PBS -A ntnu603
#PBS -l walltime=00:50:00
#PBS -l select=5:ncpus=32:mpiprocs=8:ompthreads=2
 
module load mpt
module load intelcomp
 
cd $PBS_O_WORKDIR

bash exercise2_convergence.sh
echo "completed exercise2_convergence.sh"

bash exercise2.sh
echo "completed exercise2.sh"

bash exercise3.sh
echo "completed exercise3.sh"

bash exercise4.sh
echo "completed exercise4.sh"

bash exercise5.sh
echo "completed exercise5.sh"


#!/bin/bash
#PBS -N poisson 
#PBS -A ntnu603
#PBS -l walltime=00:01:00
#PBS -l select=2:ncpus=32:mpiprocs=36
  
module load mpt
 
cd $PBS_O_WORKDIR

exercise2_convergence.sh
exercise2.sh
exercise3.sh
exercise4.sh

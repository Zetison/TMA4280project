#!/bin/bash
#PBS -N poisson 
#PBS -A ntnu603
#PBS -l walltime=00:00:09
#PBS -l select=2:ncpus=32:mpiprocs=18:ompthreads=1
  
cd $PBS_O_WORKDIR

module load mpt
module load intelcomp

mpiexec_mpt -np 1 omplace -nt 2 ./build/poisson 7 0 0 0

mpiexec_mpt -np 1 omplace -nt 4 ./build/poisson 7 0 0 0

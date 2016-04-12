#!/bin/bash
#PBS -N poisson 
#PBS -A ntnu603
#PBS -l walltime=00:00:09
#PBS -l select=1:ncpus=32:mpiprocs=4:ompthreads=4
 
module load mpt
module load intelcomp
 
cd $PBS_O_WORKDIR

mpiexec_mpt -n 4 omplace -nt 4 ./build/poisson 9 0 0 0


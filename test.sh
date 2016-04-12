#!/bin/bash
#PBS -N poisson 
#PBS -A ntnu603
#PBS -l walltime=00:00:09
#PBS -l select=1:ncpus=32:mpiprocs=4:ompthreads=4
 
module load mpt
module load intelcomp 

export MPI_DSM_VERBOSE
export OMP_NUM_THREADS 4
export KMP_AFFINITY disabled
 
cd $PBS_O_WORKDIR

mpiexec_mpt -n 4 omplace -nt 4 -c 0,1,3,4,6,7,9,10 -vv ./build/poisson 9 0 0 0


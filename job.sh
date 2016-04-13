#!/bin/bash
#PBS -N poisson 
#PBS -A ntnu603
#PBS -l walltime=00:15:00
#PBS -l select=18:ncpus=32:mpiprocs=2:ompthreads=8
 
module load mpt
module load intelcomp
export MPI_DSM_VERBOSE

cd $PBS_O_WORKDIR

bash exercise2_convergence.sh
echo "completed exercise2_convergence.sh\n\n"

bash exercise2.sh
echo "completed exercise2.sh\n\n"

bash exercise3.sh
echo "completed exercise3.sh\n\n"

bash exercise4.sh
echo "completed exercise4.sh\n\n"

bash exercise5.sh
echo "completed exercise5.sh\n\n"

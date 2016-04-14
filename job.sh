#!/bin/bash
#PBS -N poisson 
#PBS -A ntnu603
#PBS -l walltime=00:15:00
#PBS -l select=18:ncpus=32:mpiprocs=2:ompthreads=8
 
module load mpt/2.13
module load intelcomp
export MPI_DSM_VERBOSE=1

cd $PBS_O_WORKDIR

######################################################################################
echo "Exercise 2 - convergence plot"

k_start=3
k_end=16
rhsType=1
P=36
for t in 4 8; do
	for k in `seq $k_start $k_end`; do
		printf "t = "$t", k = "$k"\n"
		mpiexec_mpt -np $P omplace -nt $t -vv ./build/poisson $k $rhsType 0 1 
	done
done



######################################################################################
echo "Exercise 2"

rhsType=1
k=12

for P in 1 2 4 8 16 32; do
	for t in 1 2 4 8; do
		printf "P = "$P", t = "$t"\n"
		mpiexec_mpt -np $P omplace -nt $t -vv ./build/poisson $k $rhsType 0 0 
	done
done

######################################################################################
echo "Exercise 3"

rhsType=1
k=14
arr=(1 2 3 4 6 9 12 18 36)
N_arr=${#arr[@]}
for i in `seq 5 $N_arr`; do
	P=${arr[$[i-1]]}
	t=${arr[$[N_arr-i]]}
	printf "P = "$P", t = "$t"\n"
	mpiexec_mpt -np $P omplace -nt $t -vv ./build/poisson $k $rhsType 0 0 
done

######################################################################################
echo "Exercise 4"

rhsType=1
k_start=10
k_end=14
t=8

for P in 1 2 4 8 16 32; do
	for k in `seq $k_start $k_end`; do
		printf "P = "$P", k = "$k"\n"	
		mpiexec_mpt -np $P omplace -nt $t -vv ./build/poisson $k $rhsType 0 0 
	done
done

######################################################################################
echo "Exercise 5"

mkdir -p postProcessing
rhsType=3
k=11
t=8
P=36

cd build
mpiexec_mpt -np $P omplace -nt $t -vv ./poisson $k $rhsType 1 0


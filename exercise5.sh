#! /bin/sh

mkdir -p postProcessing
rhsType=3
k=11
t=2
P=18

export OMP_NUM_THREADS=$t
cd build
mpiexec_mpt -np $P omplace -nt $t ./poisson $k $rhsType 1 0

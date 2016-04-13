#! /bin/sh

mkdir -p postProcessing
rhsType=3
k=11
t=8
P=36

cd build
mpiexec_mpt -np $P omplace -nt $t -vv ./poisson $k $rhsType 1 0

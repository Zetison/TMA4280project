#! /bin/bash

mkdir build
cd build

module load intelcomp
module load cmake
module load mpt
CC=mpicc cmake .. -DCMAKE_BUILD_TYPE=Release
make

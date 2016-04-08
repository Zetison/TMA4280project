#! /bin/sh

mkdir build
cd build

module load intelcomp
module load cmake
CC=mpicc cmake .. -DCMAKE_BUILD_TYPE=Release
make

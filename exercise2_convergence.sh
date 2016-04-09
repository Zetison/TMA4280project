#! /bin/sh

mkdir -p results
k_start=3
k_end=14
rhsType=1
P=20
for t in 1 2; do
	filename="results/convergence_plot.txt"
	tempString="h maxRelativeError\n"
	printf "$tempString"
	printf "$tempString" > $filename


	for k in `seq $k_start $k_end`; do
		export OMP_NUM_THREADS=$t
		mpiexec_mpt -n $P ./build/poisson $k $rhsType 0 1 > temp.txt
		
		tempString="$(sed -n 4,4p temp.txt)\n"
		printf "$tempString"
		printf "$tempString" >> $filename
	done
done
rm temp.txt

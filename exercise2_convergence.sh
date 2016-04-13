#! /bin/sh

mkdir -p results
k_start=3
k_end=14
rhsType=1
P=36
for t in 4 8; do
	filename="results/convergence_plot_t"$t".dat"
	tempString="h maxRelativeError\n"
	printf "$tempString" > $filename


	for k in `seq $k_start $k_end`; do
		mpiexec_mpt -np $P omplace -nt $t -vv./build/poisson $k $rhsType 0 1 > temp.txt
		
		tempString="$(sed -n 4,4p temp.txt)\n"
		printf "$tempString" >> $filename
	done
done
rm temp.txt

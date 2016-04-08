#! /bin/sh
# The first input argument is k_start, the second is k_end and
# the third input number is the number of processes used in the MPI implementation

mkdir -p results
k_start=3
k_end=13
rhsType=2

filename="results/convergence_plot.txt"
> $filename
P=20
t=20

for (( k=$k_start; k<=$k_end; k++ )); do
	export OMP_NUM_THREADS=$t
	mpiexec -n $P ./build/poisson $k $rhsType 0 > temp.txt
	
	#print result to file (include header if k = k_start)
	if [ $k == $k_start ]; then
		tempString="$(sed -n 3,4p temp.txt)\n"
	else
		tempString="$(sed -n 4,4p temp.txt)\n"
	fi

	printf "$tempString"
	printf "$tempString" >> $filename
done
k=12
for t in 1 2 4 8 16; do
	filename="results/k"$k"_t"$t".txt"
	tempString="P time\n"
	printf "$tempString"
	printf "$tempString" >> $filename

	for P in 1 2 4 8 16; do
		export OMP_NUM_THREADS=$t
		mpiexec -n $P ./build/poisson $k $rhsType 0 > temp.txt
		tempString="$(sed -n 1,1p temp.txt)"
		tempString="${tempString##* }"
		tempString=$P" "$tempString"\n"
		printf "$tempString"
		printf "$tempString" >> $filename

	done
done





rm temp.txt

#! /bin/sh

mkdir -p results
rhsType=1
k_start=10
k_end=14
t=2
for k in `seq $k_start $k_end`; do
	P=1
	export OMP_NUM_THREADS=$t
	mpiexec -n $P ./build/poisson $k $rhsType 0 0 > temp.txt

	tempString="$(sed -n 1,1p temp.txt)"
	initialTime="${tempString##* }"
	echo "initialTime = "$initialTime
	filename="results/exercise4_k"$k".txt"
	tempString="P initialTime time\n"
	printf "$tempString"
	printf "$tempString" > $filename


	for P in 2 4 8 16; do
		export OMP_NUM_THREADS=$t
		mpiexec -n $P ./build/poisson $k $rhsType 0 0 > temp.txt
	
		tempString="$(sed -n 1,1p temp.txt)\n"
		tempString="${tempString##* }"
		tempString=$P" "$initialTime" "$tempString
		printf "$tempString"
		printf "$tempString" >> $filename
	done
done

rm temp.txt

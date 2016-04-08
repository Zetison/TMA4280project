#! /bin/sh

mkdir -p results
rhsType=2
k=14
tempString="P time\n"
filename="results/exercise3.txt"
printf "$tempString"
printf "$tempString" > $filename

arr=(1 2 3 6 9 18)
N_arr=${#arr[@]}
for i in `seq 1 $N_arr`; do
	P=${arr[$[i-1]]}
	t=${arr[$[N_arr-i]]}

	export OMP_NUM_THREADS=$t
	mpiexec -n $P ./build/poisson $k $rhsType 0 > temp.txt
	tempString="$(sed -n 1,1p temp.txt)"
	tempString="${tempString##* }"
	tempString=$P" "$tempString"\n"
	printf "$tempString"
	printf "$tempString" >> $filename
done

rm temp.txt

#! /bin/sh

mkdir -p results
rhsType=1
k_start=10
k_end=14
for k in `seq $k_start $k_end`; do

	tempString="P time\n"
	filename="results/exercise3_k"$k".dat"
	printf "$tempString" > $filename
	
	arr=(1 2 3 4 6 9 12 18 36)
	N_arr=${#arr[@]}
	for i in `seq 5 $N_arr`; do
		P=${arr[$[i-1]]}
		t=${arr[$[N_arr-i]]}
	
		mpiexec_mpt -np $P omplace -nt $t -vv ./build/poisson $k $rhsType 0 0 > temp.txt
		tempString="$(sed -n 1,1p temp.txt)"
		tempString="${tempString##* }"
		tempString=$P" "$tempString"\n"
		printf "$tempString" >> $filename
	done
done

rm temp.txt

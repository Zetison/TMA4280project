#! /bin/sh

mkdir -p results
rhsType=1
k=12
for t in 1 2 4 8 16; do
	filename="results/exercise2_k"$k"_t"$t".txt"
	tempString="P time\n"
	printf "$tempString"
	printf "$tempString" > $filename

	for P in 1 2 4 8 16; do
		export OMP_NUM_THREADS=$t
		mpiexec_mpt -np $P omplace -nt $t ./build/poisson $k $rhsType 0 0 > temp.txt
		tempString="$(sed -n 1,1p temp.txt)"
		tempString="${tempString##* }"
		tempString=$P" "$tempString"\n"
		printf "$tempString"
		printf "$tempString" >> $filename

	done
done

rm temp.txt

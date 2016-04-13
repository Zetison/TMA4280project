#! /bin/sh

mkdir -p results
rhsType=1
k_start=10
k_end=14
t=8
filename="results/exercise4.dat"
printf "P\\k" > $filename
for k in `seq $k_start $k_end`; do
	printf " "$k >> $filename
done

for P in 1 2 4 8 16 32; do
	tempString="\n"$P
	printf "$tempString" >> $filename
	for k in `seq $k_start $k_end`; do
#mpiexec -np $P ./build/poisson $k $rhsType 0 0 > temp.txt
		mpiexec_mpt -np $P omplace -nt $t -vv ./build/poisson $k $rhsType 0 0 > temp.txt

		tempString="$(sed -n 1,1p temp.txt)"
		tempString=" ${tempString##* }"
		printf "$tempString" >> $filename
	done
done

rm temp.txt

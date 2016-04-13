#! /bin/sh

mkdir -p results
rhsType=1
k=12
filename="results/exercise2.dat"
printf "P\\t" > $filename
for t in 1 2 4 8; do
	printf " "$t >> $filename
done

for P in 1 2 4 8 16 32; do
	tempString="\n"$P
	printf "$tempString" >> $filename
	for t in 1 2 4 8; do
#mpiexec -np $P ./build/poisson $k $rhsType 0 0 > temp.txt
		mpiexec_mpt -np $P omplace -nt $t -vv ./build/poisson $k $rhsType 0 0 > temp.txt

		tempString="$(sed -n 1,1p temp.txt)"
		tempString=" ${tempString##* }"
		printf "$tempString" >> $filename
	done
done

rm temp.txt

# TMA4280project
The program is compiled by running

bash compileProgram.sh

Usage for poisson program:


Usage:
  ./poisson k, rhsType, postProcessing, computeError
Arguments:
  k: the problem size n=2^k
  rhsType: choose from 0 to 3
  postProcessing: 0 or 1
  computeError: 0 or 1


The poisson program is implemented with four rhs types. The different choises
for rhsType is:
	1. f(x,y) = 2 * (y - y*y + x - x*x)
	2. f(x,y) = 5*PI'PI*sin(PI*x)*sin(2*PI*x)
	3. f(x,y) = 1
	4. f(x,y) = "three point sources"

It should be noted that since the implemented exact solution for rhsType 3 and 4
is based on the infinite series representation which has limited convergence
rate (escpecially rhsType 4), the accuracy has been limited to approximately a
thousand terms, which only yields an accuracy of 1e-8 for rhsType 3 and 1e-4.



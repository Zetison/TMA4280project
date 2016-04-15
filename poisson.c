/**
 * C program to solve the two-dimensional Poisson equation on
 * a unit square using one-dimensional eigenvalue decompositions
 * and fast sine transforms.
 *
 * Einar M. Rønquist
 * NTNU, October 2000
 * Revised, October 2001
 * Revised by Eivind Fonn, February 2015
 * Revised by Jon Vegard Venås and Morten Vassvik spring 2015
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <mpi.h> 	//needed for MPI functions
#include <omp.h>

#define PI 3.14159265358979323846
#define true 1
#define false 0
typedef int bool;
enum rhsTypes {RHS_TYPE_POLYNOMIAL, RHS_TYPE_SINE, RHS_TYPE_CONST, RHS_TYPE_POINTSOURCES, RHS_TYPE_SINCOS};

// Function prototypes
double *mk_1D_array(int n);
double **mk_2D_array(int n1, int n2);
double complex **mk_2D_array_complex(int n1, int n2);
int loc_to_glob(int i, int rank, int m, int nprocs);
double rhs(double x, double y, int rhsType, int n);
double exact_solution(double x, double y, int rhsType);
void transpose(double **A, double **At, int np, int m, int nprocs, double *recvbuf, double *sendbuf, int *sendcounts, int *sdispls, int *np_arr);
double compute_max_relative_error(double **b, int rank, int m, int np, int nprocs, double *grid_x, double *grid_y, int rhsType);
void print_matrix_to_file(double **A, int np, int m, int rank, int nprocs, int rhsType);
void fft(double complex *z, int m);
void fast_sine(double *v, int n, double complex *z, bool inverse);
void fstinv(double *v, int n, double complex *z);
void fst(double *v, int n, double complex *z);

int main(int argc, char **argv) {
	double start_time, end_time, total_time;
	start_time = MPI_Wtime();

	int nprocs, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);


	if (argc < 5) {
		if (rank == 0) {
			printf("Usage:\n");
			printf("  ./poisson k, rhsType, postProcessing, computeError\n\n");
			printf("Arguments:\n");
			printf("  k: the problem size n=2^k\n");
			printf("  rhsType: choose from 0 to 4\n");
			printf("  postProcessing: 0 or 1\n");
			printf("  computeError: 0 or 1\n\n");
		}
		MPI_Finalize();

		return -1;
	}

	int rhsType = atoi(argv[2]);
	bool postProcessing = atoi(argv[3]);
	bool computeError = atoi(argv[4]);

	// The number of grid points in each direction is n+1
	// The number of degrees of freedom in each direction is n-1
	int n = 1 << atoi(argv[1]);

	int m = n - 1;
	
	int offset = m%nprocs; // number of elements left over after evenly distribution
	int np = m/nprocs + (offset > rank ? 1 : 0);
	int tag = 1;

	double L_x, L_y;
	if (rhsType == RHS_TYPE_SINCOS) {
		L_x = 2.3;
		L_y = 0.78;
	} else {
		L_x = 1.0;
		L_y = 1.0;
	}

	double h_x = L_x/n;
	double h_y = L_y/n;
	double gamma2 = h_y*h_y/h_x/h_x;
	double **b_p = mk_2D_array(np, m);
	double **bt_p = mk_2D_array(np, m);
	
	// Initialize arrays for the transpose operation
	double *recvbuf = mk_1D_array(np*m);
	double *sendbuf = mk_1D_array(np*m);
	int *sendcounts = (int *)malloc(nprocs * sizeof(int));
	int *sdispls = (int *)malloc(nprocs * sizeof(int));
	int *np_arr = (int *)malloc(nprocs * sizeof(int));

	int displs = 0;
	for (int k = 0; k < nprocs; k++) {
		int np_k = k < offset ? m/nprocs+1 : m/nprocs;
		np_arr[k] = np_k;
		sendcounts[k] = np_k*np;
		sdispls[k] = displs;
		displs += np_k*np;
	}


	int t; // number of threads
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < 1; i++)
		t = omp_get_num_threads();

	double complex **z = mk_2D_array_complex(t, 2*n);

	// Grid points
	double *grid_x = mk_1D_array(n+1);
	double *grid_y = mk_1D_array(n+1);

	#pragma omp parallel for schedule(static)
	for (int i = 0; i < n+1; i++)
		grid_x[i] = i * h_x;

	#pragma omp parallel for schedule(static)
	for (int j = 0; j < n+1; j++)
		grid_y[j] = j * h_y;

	// The diagonal of the eigenvalue matrix of T
	double *diag = mk_1D_array(m);
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < m; i++)
		diag[i] = 2*(1 - cos((i+1)*PI/n));
	
	#pragma omp parallel for schedule(static)
	for (int j = 0; j < np; j++) {
		int j_glob = loc_to_glob(j, rank, m, nprocs);
		for (int i = 0; i < m; i++){
			double x = grid_x[i+1];
			double y = grid_y[j_glob+1];
			b_p[j][i] = h_y*h_y*rhs(x, y, rhsType, n);
			
			if(rhsType == RHS_TYPE_SINCOS){
				// Accounting for inhomogeneous Dirichlet boundary conditions
				if(j_glob == 0)
					b_p[j][i] += gamma2*exact_solution(0,y,rhsType);
				if(j_glob == m-1)
					b_p[j][i] += gamma2*exact_solution(L_x,y,rhsType);
				if(i == 0)
					b_p[j][i] += exact_solution(x,0,rhsType);
				if(i == m-1)
					b_p[j][i] += exact_solution(x,L_y,rhsType);
			}
		}
	}

	// Calculate Btilde^T = S^-1 * (S * B)^T
	// At this point b_p = B
	#pragma omp parallel for schedule(static)
	for (int j = 0; j < np; j++)
		fst(b_p[j], n, z[omp_get_thread_num()]);

	transpose(b_p, bt_p, np, m, nprocs, recvbuf, sendbuf, sendcounts, sdispls, np_arr);

	// At this point bt_p = (S B)^T
	#pragma omp parallel for schedule(static)
	for (int j = 0; j < np; j++)
		fstinv(bt_p[j], n, z[omp_get_thread_num()]);

	// solve gamma^2*Lambda * Utilde + Utilde * Lambda = Btilde
	// At this point bt_p = Btilde^T
	#pragma omp parallel for schedule(static)
	for (int j = 0; j < np; j++) {
		int j_glob = loc_to_glob(j, rank, m, nprocs);
		for (int i = 0; i < m; i++)
			bt_p[j][i] = bt_p[j][i]/(gamma2*diag[j_glob] + diag[i]);
	}

	// Calculate U = S^-1 * (S * Utilde^T)^T
	// At this point bt_p = Utilde^T
	#pragma omp parallel for schedule(static)
	for (int j = 0; j < np; j++)
		fst(bt_p[j], n, z[omp_get_thread_num()]);

	transpose(bt_p, b_p, np, m, nprocs, recvbuf, sendbuf, sendcounts, sdispls, np_arr);
	
	// At this point b_p = (S * Utilde^T)^T
	#pragma omp parallel for schedule(static)
	for (int j = 0; j < np; j++)
		fstinv(b_p[j], n, z[omp_get_thread_num()]);
	
	end_time = MPI_Wtime();
	total_time = end_time-start_time;
	
	if (rank == 0)
		printf("Time elapsed: %e\n", total_time);

	if (postProcessing)
		print_matrix_to_file(b_p, np, m, rank, nprocs, rhsType);

	if (computeError) {
		double maxRelativeError = compute_max_relative_error(b_p, rank, m, np, nprocs, grid_x, grid_y, rhsType);
		if (rank == 0) {
			printf("\n%21s%21s\n", "h", "maxRelativeError");
			printf("%21.17f%21.17f\n", h_x > h_y ? h_x : h_y , maxRelativeError);
		}
	}
	
	MPI_Finalize();

	return 0;
}

double *mk_1D_array(int n) {
	return (double *)malloc(n * sizeof(double));
}

double **mk_2D_array(int n1, int n2) {
	double **ret = (double **)malloc(n1 * sizeof(double *));
	
	ret[0] = (double *)malloc(n1 * n2 * sizeof(double));

	for (int j = 1; j < n1; j++)
		ret[j] = ret[j-1] + n2;
	return ret;
}

double complex **mk_2D_array_complex(int n1, int n2) {
	double complex **ret = (double complex **)malloc(n1 * sizeof(double complex *));
	
	ret[0] = (double complex*)malloc(n1 * n2 * sizeof(double complex));
   
	for (int j = 1; j < n1; j++)
		ret[j] = ret[j-1] + n2;
	return ret;
}

int loc_to_glob(int j, int rank, int m, int nprocs) {
	int offset = m%nprocs;
	return rank*(m/nprocs) + (offset > rank ? rank : offset) + j;
}

double rhs(double x, double y, int rhsType, int n) {
	if (rhsType == RHS_TYPE_POLYNOMIAL)
		return 2 * (y - y*y + x - x*x);
	else if (rhsType == RHS_TYPE_SINE)
		return 5*PI*PI*sin(PI*x)*sin(2*PI*y);
	else if (rhsType == RHS_TYPE_CONST)
		return 1.0;
	else if (rhsType == RHS_TYPE_POINTSOURCES){
		if (x == 0.5 && y == 0.5 || x == 0.75 && y == 0.75)
			return n*n;
		else if (x == 0.25 && y == 0.5)
			return -2*n*n;
		else
			return 0.0;
	} else if (rhsType == RHS_TYPE_SINCOS)
		return 125*sin(10*x)*cos(5*y);
	else
		exit(EXIT_FAILURE);
	return 2 * (y - y*y + x - x*x);
}

double exact_solution(double x, double y, int rhsType) {
	if (rhsType == RHS_TYPE_POLYNOMIAL)
		return x*(x-1)*y*(y-1);
	else if (rhsType == RHS_TYPE_SINE)
		return sin(PI*x)*sin(2*PI*y);
	else if (rhsType == RHS_TYPE_CONST || rhsType == RHS_TYPE_POINTSOURCES) {
		if (rhsType == RHS_TYPE_POINTSOURCES && ((x == 0.5 && y == 0.5) || (x == 0.75 && y == 0.75) || (x == 0.25 && y == 0.5)))
			return NAN;

		double ratio, u = 0;
		#pragma omp parallel for reduction(+:u) 
		for (int N = 0; N < 1000; N++) {
			double maxTerm = 0, a_mn;
			int n = N;
			for (int m = 0; m <= N; m++) {
				double fact_x, fact_y;
				if (rhsType == 3) {
					fact_x = (2*m+1)*PI;
				fact_y = (2*n+1)*PI;
					a_mn = 4/PI/PI/(4*m*n+2*m+2*n+1);
				} else if (rhsType == 4) {
					fact_x = (m+1)*PI;
					fact_y = (n+1)*PI;
					double Q[3] = {1, 1, -2};
					double pts[3][2] = {{0.5, 0.5},
							    {0.75, 0.75},
							    {0.25, 0.5}};
					a_mn = 0;
					for (int i = 0; i < 3; i++)
						a_mn += Q[i]*sin(fact_x*pts[i][0])*sin(fact_y*pts[i][1]);
				}
				a_mn *= 4/(fact_x*fact_x+fact_y*fact_y);
				double uTerm = a_mn*sin(fact_x*x)*sin(fact_y*y);

				u += uTerm;
			n--;
			}
		}
		return u;
	} else if (rhsType == RHS_TYPE_SINCOS)
		return sin(10*x)*cos(5*y);
	else
		exit(EXIT_FAILURE);
}



void transpose(double **A, double **At, int np, int m, int nprocs, double *recvbuf, double *sendbuf, int *sendcounts, int *sdispls, int *np_arr) {
	int counter = 0;
	for (int k = 0; k < nprocs; k++) {
		int np_k = np_arr[k];
		for (int j = 0; j < np; j++)
			for (int i = 0; i < np_k; i++)
				sendbuf[counter++] = A[j][loc_to_glob(i, k, m, nprocs)];
	}
	MPI_Alltoallv(sendbuf, sendcounts, sdispls,  MPI_DOUBLE, recvbuf, sendcounts, sdispls, MPI_DOUBLE, MPI_COMM_WORLD);
	
	counter = 0;
	for (int k = 0; k < nprocs; k++) {
		int np_k = np_arr[k];
		for (int i = 0; i < np_k; i++){
			int i_glob = loc_to_glob(i, k, m, nprocs);
			for (int j = 0; j < np; j++)
				At[j][i_glob] = recvbuf[counter++];
		}
	}
}

double compute_max_relative_error(double **b, int rank, int m, int np, int nprocs, double *grid_x, double *grid_y, int rhsType) {
	double max_u = -1;
	double max_error = -1;

	for (int j = 0; j < np; j++) {
		int j_glob = loc_to_glob(j, rank, m, nprocs);
		for (int i = 0; i < m; i++) {
			double u = exact_solution(grid_x[i+1], grid_y[j_glob+1], rhsType);
			if (max_u < fabs(u))
				max_u = fabs(u);
			double error;
			if (isnan(u))
				error = 0;
			else
				error = fabs(u-b[j][i]);
			if (max_error < error)
				max_error = error;
		}
	}
	double global_max_error, global_max_u;

	MPI_Reduce(&max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&max_u, &global_max_u, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	return global_max_error/global_max_u;
}

void print_matrix(double *A, int np, int m, FILE *fp) {
	for (int j = 0; j < np; j++) {
		for (int i = 0; i < m; i++)
			fprintf(fp, "%20.15g ", A[i+j*m]);
		fprintf(fp, "\n");
	}
}

void print_matrix_to_file(double **A, int np, int m, int rank, int nprocs, int rhsType) {
	int tag = 1;
	
	if (rank == 0) {
		MPI_Status status;
		int offset = m%nprocs;
		FILE *fp;

		char fileName[80];
		sprintf(fileName, "../postProcessing/rhs%d_m%d.dat", rhsType, m);
		fp = fopen(fileName, "w+");

		print_matrix(A[0], np, m, fp);
		int np_max = m/nprocs + 1;
		double *A_k = mk_1D_array(np_max*m);
		for (int k = 1; k < nprocs; k++) {
			int np_k = m/nprocs + (offset > k ? 1 : 0);
			MPI_Recv(A_k, np_k*m, MPI_DOUBLE, k, tag, MPI_COMM_WORLD, &status);
			print_matrix(A_k, np_k, m, fp);
		}
		free(A_k);
		fclose(fp);
	} else
		MPI_Send(A[0], np*m, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
}

void fft(double complex *z, int m) {
	for (int index = 0; index < m; index++) {
		int logcounter = m;
		int rindex = 0;
		int q = index;

		do {
			int s = q / 2;
			rindex = (2 * rindex) + (q - (2 * s));
			q = s;
			logcounter = logcounter / 2;
		} while (logcounter > 1);

		if (rindex > index) {
			double complex temp = z[index];
			z[index] = z[rindex];
			z[rindex] = temp;
		}
	}

	int halflevel = 1;
	int level = 2;
	do {
		int cols = m / level;
		for (int k = 0; k < cols; k++) {
			for (int j = 0; j < halflevel; j++) {
				double complex omega = cos(2.0*PI*j / level) -1.0*sin(2.0*PI*j / level)*I;
				double complex ctemp = omega*z[(k*level) + j + halflevel];
				z[(k*level) + j + halflevel] = z[(k*level) + j] - ctemp;
				z[(k*level) + j] = z[(k*level) + j] + ctemp;
			}
		}

		halflevel = level;
		level = level * 2;
	} while (level <= m);
}

void fast_sine(double *v, int n, double complex *z, bool inverse) {
	int kk = 0;
	z[kk++] = 0;
	for (int k = 0; k < n - 1; k++)
		z[kk++] = v[k];
	
	z[kk++] = 0;
	for (int k = n - 2; k >= 0; k--)
		z[kk++] = -v[k];
	
	fft(z, 2 * n);

	for (int i = 0; i < 2 * n; i++)
		z[i] = z[i] * I/2;

	for (int i = 0; i < n - 1; i++)
		v[i] = creal(z[i + 1]);
	
	if (inverse)
		for (int i = 0; i < n - 1; i++)
			v[i] = (2.0*v[i]) / n;
}

void fstinv(double *v, int n, double complex *z) {
	fast_sine(v, n, z, false);
}

void fst(double *v, int n, double complex *z) {
	fast_sine(v, n, z, true);
}


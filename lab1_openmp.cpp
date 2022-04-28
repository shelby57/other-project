#include <cmath>
#include <iomanip>
#include <iostream>
#include <omp.h>

//#pragma GCC optimize("O3")
using namespace std;
const int M = 7;
const double eps = 1e-6;
int size, N, num_threads;
const int block = 1000;
double SpMv_time = 0.0;
double axpby_time = 0.0;
double dot_time = 0.0;

void generation_matrix(int &Nx, int &Ny, int &Nz, int *col, double *val) {
	for (int i = 0; i < size; ++i)
		col[i] = -1;
	int i = 0;
	for (int z = 0; z < Nz; ++z)
		for (int y = 0; y < Ny; ++y)
			for (int x = 0; x < Nx; ++x) {
				int ind = i * M;
				if (z > 0)
					col[ind] = i - Nx * Ny;
				if (y > 0)
					col[ind + 1] = i - Nx;
				if (x > 0)
					col[ind + 2] = i - 1;
				col[ind + 3] = i;
				if (x < Nx - 1)
					col[ind + 4] = i + 1;
				if (y < Ny - 1)
					col[ind + 5] = i + Nx;
				if (z < Nz - 1)
					col[ind + 6] = i + Nx * Ny;
				++i;
			}
	for (i = 0; i < N; ++i) {
		double sum = 0;
		for (int p = 0; p < M; ++p) {
			int j = col[i * M + p];
			if (i != j && j != -1) {
				val[i * M + p] = cos(i * j + 3.14);
				sum += abs(val[i * M + p]);
			}
		}
		val[i * M + 3] = 1.5 * sum;
	}
}

void SpMv(double *ans, const int *col, const double *val, const double *x) {
	double start = omp_get_wtime();
	#pragma omp parallel for schedule(static, 1000)
	for (int i = 0; i < N; ++i) {
		ans[i] = 0;
		int ind = i * 7;
		for (int j = 0; j < M; ++j) {
			if (col[ind + j] != -1)
				ans[i] += x[col[ind + j]] * val[ind + j];
		}
	}
	SpMv_time += omp_get_wtime() - start;
}

void axpby(double *ans, double *first, double *second, const double kf2) {
	double start = omp_get_wtime();
	#pragma omp parallel for schedule(static, 1000)
	for (int i = 0; i < N; ++i)
		ans[i] = first[i] + second[i] * kf2;
	axpby_time += omp_get_wtime() - start;
}


double dot(const double *first, const double *second) {
	double ans = 0.0;
	double start = omp_get_wtime();
	#pragma omp parallel for reduction(+:ans) schedule(static, 1000)
	for (int i = 0; i < N; ++i)
		ans += first[i] * second[i];
	dot_time += omp_get_wtime() - start;
	return ans;
}

void get_b(double *b) {
	for (int i = 0; i < N; ++i)
		b[i] = cos(i);
}

int main() {
	int Nx = 1000, Ny = 1000, Nz = 1;
	N = Nx * Ny * Nz;
	size = N * M;
	int *col = (int *) malloc(sizeof(int) * size);
	double *b = (double *) malloc(sizeof(double) * N);
	double *val = (double *) malloc(sizeof(double) * size);
	omp_set_num_threads(4);
	generation_matrix(Nx, Ny, Nz, col, val);
	get_b(b);
	
	double *q = (double *) malloc(sizeof(double) * N);
	double *r = (double *) malloc(sizeof(double) * N);
	double *p = (double *) malloc(sizeof(double) * N);
	double *x = (double *) malloc(sizeof(double) * N);
	double prev_c, c;
	double start_time = omp_get_wtime();
	for (int i = 0; i < N; ++i)
		x[i] = 0;
	bool convergence = false;
	int k = 1;
	SpMv(q, col, val, x);
	axpby(r, b, q, -1);
	do {
		c = dot(r, r);
		if (k == 1) {
			axpby(p, r, q, 0);
		} else {
			double beta = c / prev_c;
			axpby(p, r, p, beta);
		}
		SpMv(q, col, val, p);
		double alpha = c / dot(p, q);
		axpby(x, x, p, alpha);
		axpby(r, r, q, -alpha);
		if (c < eps || k >= N)
			convergence = true;
		else
			++k;
		prev_c = c;
	} while (!convergence);
	double end_time = omp_get_wtime();
	cout << SpMv_time * 1000 << " mc\n";
	cout << axpby_time * 1000 << " mc\n";
	cout << dot_time * 1000 << " mc\n";
	cout << (end_time - start_time) * 1000 << " mc\n";
	free(val);
	free(col);
	free(b);

	free(x);
	free(r);
	free(q);
	free(p);
	return 0;
}


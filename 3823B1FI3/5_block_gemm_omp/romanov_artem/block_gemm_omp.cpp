#pragma GCC optimize("O3")
#pragma GCC optimize("fast-math")

#include "block_gemm_omp.h"

#include <omp.h>
#include <vector>

inline void swap(float& a, float& b) {
	float c = b;
	b = a;
	a = c;
}

#pragma GCC target("avx2")
void MatrixProduct(const float* A, const float* B, float* res, int n) {
    
    constexpr int BLOCK_SIZE = 32;
	int TOTAL_BLOCKS = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

	float* bt = new float[n * n];

#pragma omp parallel for
	for (int i = 0; i < n * n; ++i) {
		bt[i] = B[i];
	}

#pragma omp parallel for
	for (int i = 0; i < n; ++i) {
		for (int j = i + 1; j < n; ++j) {
			swap(bt[i * n + j], bt[j * n + i]);
		}
	}

	const float* BT = bt;

	int TOTAL_BLOCKS_MINUS_ONE = TOTAL_BLOCKS - 1;

#pragma omp parallel for
	for (int pi = 0; pi < TOTAL_BLOCKS_MINUS_ONE; ++pi) {
		int li = pi * BLOCK_SIZE;
		int ri = li + BLOCK_SIZE;
		for (int pj = 0; pj < TOTAL_BLOCKS_MINUS_ONE; ++pj) {
			int lj = pj * BLOCK_SIZE;
			int rj = lj + BLOCK_SIZE;
			for (int pk = 0; pk < TOTAL_BLOCKS_MINUS_ONE; ++pk) {
				int lk = pk * BLOCK_SIZE;
				int rk = lk + BLOCK_SIZE;
				for (int i = li; i < ri; ++i) {
					for (int j = lj; j < rj; ++j) {
						float temp = 0.0f;
						const float* Ai  = A  + i * n;
						const float* BTj = BT + j * n;
						#pragma omp simd
						for (int k = lk; k < rk; ++k) {
							temp += Ai[k] * BTj[k];
						}
						res[i * n + j] += temp;
					}
				}
			}			
		}
	}

	int l = TOTAL_BLOCKS_MINUS_ONE * BLOCK_SIZE;
	int r = n;

#pragma omp parallel for
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			float temp = 0.0f;
			const float* Ai  = A  + i * n;
			const float* BTj = BT + j * n;
			#pragma omp simd
			for (int k = l; k < r; ++k) {
				temp += Ai[k] * BTj[k];
			}
			res[i * n + j] += temp;
		}
	}

#pragma omp parallel for
	for (int i = 0; i < l; ++i) {
		for (int j = l; j < r; ++j) {
			float temp = 0.0f;
			const float* Ai  = A  + i * n;
			const float* BTj = BT + j * n;
			#pragma omp simd
			for (int k = 0; k < l; ++k) {
				temp += Ai[k] * BTj[k];
			}
			res[i * n + j] += temp;
		}
	}

#pragma omp parallel for
	for (int i = l; i < r; ++i) {
		for (int j = l; j < r; ++j) {
			float temp = 0.0f;
			const float* Ai  = A  + i * n;
			const float* BTj = BT + j * n;
			#pragma omp simd
			for (int k = 0; k < l; ++k) {
				temp += Ai[k] * BTj[k];
			}
			res[i * n + j] += temp;
		}
	}

#pragma omp parallel for
	for (int i = l; i < r; ++i) {
		for (int j = 0; j < l; ++j) {
			float temp = 0.0f;
			const float* Ai  = A  + i * n;
			const float* BTj = BT + j * n;
			#pragma omp simd
			for (int k = 0; k < l; ++k) {
				temp += Ai[k] * BTj[k];
			}
			res[i * n + j] += temp;
		}
	}
	delete[] bt;
}

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
	
	const float* A = a.data();
	const float* B = b.data();

	std::vector<float> result(n * n, 0.0f);

	float* res = result.data();

    MatrixProduct(A, B, res, n);

	return result;
}
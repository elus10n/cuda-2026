#include "naive_gemm_omp.h"
#include "omp.h"
#include <vector>

// GEMM - general matrix multiplication

/*
Ключи в Visual Studio:
/openmp (/openmp:llvm) - многопоточность
/fp:fast вместо /fp:precise - быстрые floating point (expf становится быстрее)
/arch:AVX2 - векторные инструкции (в регистре одновременно 8 float)
AVX2 регстры: 256 бит; float: 32 бит; 256/32 = 8
/GL - глобальная оптимизация
/O2  - агрессивная оптимизация
/openmp:experimental - для simd (векторизации)
*/

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {

    // результирующая матрица
    std::vector<float> answer(n * n, 0.0f);

    // распараллеливаем внешний цикл по строкам
    // порядок i -> k -> j для cache-friendly доступа
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {

            // a[i * n + k] - не зависит от j
            // вытаскиваем из внутреннего цикла, чтобы не считать каждый раз
            float a_ik = a[i * n + k];

            // индексы для удобства
            int res_row_idx = i * n;
            int b_row_idx = k * n;

            // векторизация и развертывание цикла по j
            // идем по строке матрицы B и C последовательно
#pragma omp simd
            for (int j = 0; j < n; j += 4) {
                // разворачивание цикла на 4 для помощи компилятору
                
                answer[res_row_idx + j] += a_ik * b[b_row_idx + j];
                answer[res_row_idx + j + 1] += a_ik * b[b_row_idx + j + 1];
                answer[res_row_idx + j + 2] += a_ik * b[b_row_idx + j + 2];
                answer[res_row_idx + j + 3] += a_ik * b[b_row_idx + j + 3];
            }

            // так как n - степень двойки, остатки можно не проверять
        }
    }

    return answer;
}
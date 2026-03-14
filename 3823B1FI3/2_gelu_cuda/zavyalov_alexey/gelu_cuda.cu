#include "gelu_cuda.h"

#define sqrt_2_div_pi 0.7978845608028653558f
#define two_mult_sqrt_2_div_pi 1.5957691216057307116f

__global__ void gelu_kernel(const float* input, float* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
        result[i] = input[i] / (1.0f + (exp(-(two_mult_sqrt_2_div_pi * (input[i] + 0.044715f * input[i] * input[i] * input[i])))));
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int n = input.size();
    const int block_size = 256;
    int num_blocks = (n / 2 + block_size - 1) / block_size;

    float *a_gpu_first_half;
    cudaMalloc((void **)&a_gpu_first_half, n / 2 * sizeof(float));
    cudaMemcpy(a_gpu_first_half, input.data(), n / 2 * sizeof(float), cudaMemcpyHostToDevice);

    float *result_gpu_first_half;
    cudaMalloc((void **)&result_gpu_first_half, n * sizeof(float));


    cudaStream_t strm;
    float *a_gpu_second_half;
    cudaMallocAsync((void **)&a_gpu_second_half, (n + 1) / 2 * sizeof(float), strm); // надо сделать асинк
    cudaMemcpyAsync(a_gpu_second_half, input.data() + n / 2, (n + 1) / 2 * sizeof(float), cudaMemcpyHostToDevice, strm); // надо сделать асинк

    gelu_kernel<<<num_blocks, block_size>>>(a_gpu_first_half, result_gpu_first_half, n / 2);
    std::vector<float> res(input.size());

    cudaMemcpyAsync(res.data(), result_gpu_first_half, n / 2 * sizeof(float), cudaMemcpyDeviceToHost, strm); // надо сделать асинк


    float *result_gpu_second_half;
    cudaMallocAsync((void **)&result_gpu_second_half, (n + 1) / 2 * sizeof(float), strm);

    cudaStreamSynchronize(strm);

    gelu_kernel<<<num_blocks, block_size>>>(a_gpu_second_half, result_gpu_second_half, (n + 1) / 2);

    cudaMemcpy(res.data() + n / 2, result_gpu_second_half, (n + 1) / 2 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(a_gpu_first_half);
    cudaFree(a_gpu_second_half);
    cudaFree(result_gpu_second_half);
    cudaFree(result_gpu_first_half);

    return res;
}

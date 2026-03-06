#include "gelu_cuda.h"
#include <cuda_runtime.h>

/*
GELU(x) = 0.5 * x * (1 + tanh(z))
z = sqrt(2/pi) * (x + 0.044715 * x^3)

tanh(z) = (e^z - e^-z) / (e^z + e^-z):
1 + tanh(z) = 1 + (e^z - e^-z) / (e^z + e^-z)
            = (e^z + e^-z + e^z - e^-z) / (e^z + e^-z)
            = 2 * e^z / (e^z + e^-z)

1 + tanh(z) = 2 / (1 + e^(-2z))
GELU(x) = 0.5 * x * (2 / (1 + e^(-2z))) = x / (1 + e^(-2z))
*/

#define SQRT_2_OVER_PI 0.7978845608f
#define COEFF 0.044715f

__global__ void gelu_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        float x3 = x * x * x;
        float arg = SQRT_2_OVER_PI * (x + COEFF * x3);
        out[i] = x / (1.0f + expf(-2.0f * arg));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t n = input.size();

    size_t bytes = n * sizeof(float);
    float *in, *out;
    cudaMalloc(&in, bytes);
    cudaMalloc(&out, bytes);

    cudaMemcpy(in, input.data(), bytes, cudaMemcpyHostToDevice);

    const int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    gelu_kernel <<< num_blocks, block_size >>> (in, out, n);

    std::vector<float> output(n);
    cudaMemcpy(output.data(), out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(in);
    cudaFree(out);

    return output;
}

#include <iostream>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <math.h>

#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA错误: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

__global__ void stencil_1d_shared(float *in, float *out, int n)
{
    __shared__ float shared_mem[256 + 2];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < n) {
        shared_mem[tid + 1] = in[idx];
    }

    if (tid == 0 && idx > 0) {
        shared_mem[0] = in[idx - 1];
    }
    if (tid == blockDim.x - 1 && idx < n - 1) {
        shared_mem[blockDim.x + 1] = in[idx + 1];
    }

    __syncthreads();

    if (idx < n && idx > 0 && idx < n - 1) {
        out[idx] = (shared_mem[tid] + 2 * shared_mem[tid + 1] + shared_mem[tid + 2]) / 4.0f;
    } else if (idx == 0) {
        out[idx] = (2 * shared_mem[tid + 1] + shared_mem[tid + 2]) / 3.0f;
    } else if (idx == n - 1) {
        out[idx] = (shared_mem[tid] + 2 * shared_mem[tid + 1]) / 3.0f;
    }
}

void stencil_1d_cpu(float *in, float *out, int n)
{
    for (int i = 0; i < n; i++) {
        if (i == 0) {
            out[i] = (2 * in[i] + in[i + 1]) / 3.0f;
        } else if (i == n - 1) {
            out[i] = (in[i - 1] + 2 * in[i]) / 3.0f;
        } else {
            out[i] = (in[i - 1] + 2 * in[i] + in[i + 1]) / 4.0f;
        }
    }
}

int main()
{
    const int n = 1024;
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    float *h_in = (float*)malloc(n * sizeof(float));
    float *h_out_gpu = (float*)malloc(n * sizeof(float));
    float *h_out_cpu = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        h_in[i] = sin(2 * M_PI * i / n);
    }

    float *d_in, *d_out;
    CHECK_CUDA_ERROR(cudaMalloc(&d_in, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_out, n * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));

    stencil_1d_shared<<<grid_size, block_size>>>(d_in, d_out, n);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(h_out_gpu, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

    stencil_1d_cpu(h_in, h_out_cpu, n);

    float max_error = 0.0f;
    for (int i = 0; i < n; i++) {
        float error = fabs(h_out_gpu[i] - h_out_cpu[i]);
        max_error = max(max_error, error);
    }
    std::cout << "最大误差: " << max_error << std::endl;

    free(h_in);
    free(h_out_gpu);
    free(h_out_cpu);
    cudaFree(d_in);
    cudaFree(d_out);

    std::cout << "1维蒙板计算完成！" << std::endl;
    return 0;
}
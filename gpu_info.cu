#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("GPU型号: %s\n", prop.name);
    printf("SM数量(multiProcessorCount): %d\n", prop.multiProcessorCount);
    printf("Warp尺寸(warpSize): %d\n", prop.warpSize);
    printf("单Block最大线程(maxThreadsPerBlock): %d\n", prop.maxThreadsPerBlock);
    printf("每SM最大驻留线程(maxThreadsPerMultiProcessor): %d\n", prop.maxThreadsPerMultiProcessor);
    printf("计算能力: %d.%d\n", prop.major, prop.minor);
    return 0;
}

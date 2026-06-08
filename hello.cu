#include <iostream>
#include <cuda_runtime.h>

__global__ void add(int *a,int *b,int *c)
{
	int tid = threadIdx.x;
	c[tid] = a[tid] + b[tid];
}

int main()
{
	int a,b,c;
	int *da,*db,*dc;
	int size = sizeof(int);

	cudaMalloc(&da,size);
	cudaMalloc(&db,size);
	cudaMalloc(&dc,size);
	a=2,b=7;
	cudaMemcpy(da,&a,size,cudaMemcpyHostToDevice);
	cudaMemcpy(db,&b,size,cudaMemcpyHostToDevice);
	add<<<1,1>>>(da,db,dc);
	cudaMemcpy(&c,dc,size,cudaMemcpyDeviceToHost);
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);
	std::cout << "c=" << c << std::endl;
	return 0;
}

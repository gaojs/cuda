#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

void mul_host(float *a,float *b,float *c, int n) 
{
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			float s= 0.0f;
			for (int k = 0; k < n; k++) {
				s += a[i * n + k] * b[k * n + j];
			}
			c[i * n + j] = s;
		}
	}
}

__global__ void mul_kernel(float *a,float *b,float *c, int n) 
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	float s= 0.0f;
	for (int k = 0; k < n; k++) {
		s += a[ty * n + k] * b[k * n + tx];
	}
	c[ty * n + tx] = s;
}

void mul_cuda(float *a,float *b,float *c, int n) 
{
	int size = n*n*sizeof(float);
	float *da,*db,*dc;
	// malloc device memory
	cudaMalloc(&da,size);
	cudaMemcpy(da,a,size,cudaMemcpyHostToDevice);
	cudaMalloc(&db,size);
	cudaMemcpy(db,b,size,cudaMemcpyHostToDevice);
	cudaMalloc(&dc,size);
	// launch kernel
	dim3 gridDim(1,1);
	dim3 blockDim(n,n);
	mul_kernel<<<gridDim,blockDim>>>(da,db,dc,n);
	cudaDeviceSynchronize();
	// copy result to host memory
	cudaMemcpy(c,dc,size,cudaMemcpyDeviceToHost);
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);
}

void print(float *a, int N)
{
	std::cout << std::setprecision(0) << std::fixed;
	for(int i=0;i<N;i++)	{
		for(int j=0;j<N;j++)	{
			std::cout << a[i*N+j] << " ";
		}
		std::cout << std::endl;
	}
}

int main()
{
	const int N=4;
	// host memory
	float *a,*b,*c;
	a=(float*)malloc(N*N*sizeof(float));
	b=(float*)malloc(N*N*sizeof(float));
	c=(float*)malloc(N*N*sizeof(float));
	// initialize matrix a and b
	for(int i=0;i<N*N;i++)	{
		a[i]=rand()%10;
		b[i]=rand()%10;
		c[i]=0.0f;
	}
	print(a,N);
	print(b,N);
	mul_host(a,b,c,N);
	print(c,N);
	mul_cuda(a,b,c,N);
	print(c,N);
	// free host memory
	free(a);
	free(b);
	free(c);
	return 0;
}

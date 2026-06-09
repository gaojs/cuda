#include <iostream>
#include <cuda_runtime.h>


// 使用__global__声明CUDA核函数
__global__ void add(int *a,int *b,int *c)
{
	// int tid = threadIdx.x; // blockIdx.x;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	c[tid] = a[tid] + b[tid];
}

void random_ints(int *a,int size)
{
	for(int i=0;i<size;i++)	{
		a[i]=rand()%1000;
	}
}

const int N=512;
const int M=8;

void test_add() 
{	
	int *a,*b,*c;
	int *da,*db,*dc;
	int size = N*sizeof(int);

	cudaMalloc(&da,size);
	cudaMalloc(&db,size);
	cudaMalloc(&dc,size);
	// a=2,b=7;
	a=(int*)malloc(size);random_ints(a,N);
	b=(int*)malloc(size);random_ints(b,N);
	std::cout << "a[0]=" << a[0] << std::endl;
	std::cout << "b[0]=" << b[0] << std::endl;
	c=(int*)malloc(size);
	cudaMemcpy(da,a,size,cudaMemcpyHostToDevice);
	cudaMemcpy(db,b,size,cudaMemcpyHostToDevice);
	// add<<<1,1>>>(da,db,dc);
	// add<<<N,1>>>(da,db,dc);
	add<<<N/M,M>>>(da,db,dc);
	cudaMemcpy(c,dc,size,cudaMemcpyDeviceToHost);
	std::cout << "c[0]=" << c[0] << std::endl;
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);
	free(a);
	free(b);
	free(c);
}

__global__ void stencil_1d(int *in, int *out) 
{
	const int RADIUS=3;
	const int BLOCK_SIZE=1;
	__shared__ int temp[BLOCK_SIZE+2*RADIUS];
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int lid = RADIUS + threadIdx.x;
	temp[lid] = in[gid];
	if(threadIdx.x<RADIUS) {
		temp[lid-RADIUS] = in[gid-RADIUS];
		temp[lid+RADIUS] = in[gid+RADIUS];
	}
	__syncthreads();
	int res = 0;
	for(int i=-RADIUS;i<=RADIUS;i++) {
		res += temp[lid+i];
	}
	out[gid] = res;
}

void test_stencil_1d()
{
	int *in,*out;
	int *da,*dout;
	int size = N*sizeof(int);

	cudaMalloc(&da,size);
	cudaMalloc(&dout,size);
	in=(int*)malloc(size);random_ints(in,N);
	out=(int*)malloc(size);
	cudaMemcpy(da,in,size,cudaMemcpyHostToDevice);
	stencil_1d<<<N/M,M>>>(da,dout);
	cudaMemcpy(out,dout,size,cudaMemcpyDeviceToHost);
	std::cout << "out[0]=" << out[0] << std::endl;
	free(in);
	free(out);
	cudaFree(da);
	cudaFree(dout);
}

int main()
{
	// test_add();
	test_stencil_1d();
	return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define KERNEL_SIZE 20

__constant__ int kernel[KERNEL_SIZE];

__global__ void conv1d(int *input, int *output, int l) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ int s_input[];
	int r = KERNEL_SIZE / 2;
	int d = r * 2;
	int n_padded = blockDim.x + d;
	int offset = threadIdx.x + blockDim.x;
	int g_offset = blockDim.x * blockIdx.x + offset;
	s_input[threadIdx.x] = input[tid];

	if (offset < n_padded) {
		s_input[offset] = input[g_offset];
	}
	__syncthreads();

	int temp = 0;
	for (int j = 0; j < KERNEL_SIZE; j++) {
		temp += s_input[threadIdx.x + j] * kernel[j];
	}
	output[tid] = temp;
}

int main() {
	int l = 20480;
	int i;
	int r = KERNEL_SIZE / 2;
	int n = l + r * 2;
	int *host_input, *host_kernel, *host_output;
	int *dev_input, *dev_output;

	cudaMalloc((void**)&dev_input, sizeof(int) * n);
	cudaMalloc((void**)&dev_output, sizeof(int) * KERNEL_SIZE);
	cudaMallocHost((void**)&host_input, sizeof(int) * n);
	cudaMallocHost((void**)&host_kernel, sizeof(int) * KERNEL_SIZE);
	cudaMallocHost((void**)&host_output, sizeof(int) * l);

	for (i = 0; i < n; i++) {
		if ((i < r) || (i >= l + r)) {
			host_input[i] = 0;
		}
		else {
			host_input[i] = round(rand());
		}
	}
	for (i = 0; i < KERNEL_SIZE; i++) {
		host_kernel[i] = round(rand());
	}

	printf("Start convolution\n");
	clock_t start_time = clock();
	cudaMemcpy(dev_input, host_input, sizeof(int) * n, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(kernel, host_kernel, sizeof(int) * KERNEL_SIZE);

	int block = 256;
	int grid = (l + block - 1) / block;
	size_t sharemem = sizeof(int) * (block + r * 2);
	conv1d<<<grid, block, sharemem>>>(dev_input, dev_output, l);

	cudaMemcpy(host_output, dev_output, sizeof(int) * l, cudaMemcpyDeviceToHost);
	clock_t end_time = clock();
	printf("Time consuming of 1D convolution of %d array with %d kernel is %f ms.\n", l, KERNEL_SIZE, static_cast<double>(end_time - start_time)/CLOCKS_PER_SEC*1000);

	cudaFree(dev_input);
	cudaFree(dev_output);
	cudaFreeHost(host_input);
	cudaFreeHost(host_kernel);
	cudaFreeHost(host_output);

	return 0;
}

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
	int r = KERNEL_SIZE / 2;
	int start = tid - r;
	int temp = 0;
	for (int j = 0; j < KERNEL_SIZE; j++) {
		if ((start + j >= 0) && (start + j < l)) {
			temp += input[start + j] * kernel[j];
		}
	}
	output[tid] = temp;
}

int main() {
	int l = 20480;
	int i;
	int *host_input, *host_kernel, *host_output;
	int *dev_input, *dev_output;

	cudaMalloc((void**)&dev_input, sizeof(int) * l);
	cudaMalloc((void**)&dev_output, sizeof(int) * KERNEL_SIZE);
	cudaMallocHost((void**)&host_input, sizeof(int) * l);
	cudaMallocHost((void**)&host_kernel, sizeof(int) * KERNEL_SIZE);
	cudaMallocHost((void**)&host_output, sizeof(int) * l);

	for (i = 0; i < l; i++) {
		host_input[i] = round(rand());
	}
	for (i = 0; i < KERNEL_SIZE; i++) {
		host_kernel[i] = round(rand());
	}

	clock_t start_time = clock();
	cudaMemcpy(dev_input, host_input, sizeof(int) * l, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(kernel, host_kernel, sizeof(int) * KERNEL_SIZE);

	int block = 256;
	int grid = (l + block - 1) / block;

	conv1d<<<grid, block>>>(dev_input, dev_output, l);

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

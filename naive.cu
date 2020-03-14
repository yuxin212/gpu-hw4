#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void conv1d(int *input, int *kernel, int *output, int l, int k) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int r = k / 2;
	int start = tid - r;
	int temp = 0;
	for (int j = 0; j < k; j++) {
		if ((start + j >= 0) && (start + j < l)) {
			temp += input[start + j] * kernel[j];
		}
	}
	output[tid] = temp;
}

int main() {
	int l = 20480;
	int k = 7;
	int i;
	int *input, *kernel, *output;
	int *dev_input, *dev_kernel, *dev_output;

	cudaMalloc((void**)&dev_input, sizeof(int) * l);
	cudaMalloc((void**)&dev_kernel, sizeof(int) * k);
	cudaMalloc((void**)&dev_output, sizeof(int) * l);
	cudaMallocHost((void**)&input, sizeof(int) * l);
	cudaMallocHost((void**)&kernel, sizeof(int) * k);
	cudaMallocHost((void**)&output, sizeof(int) * l);

	for (i = 0; i < l; i++) {
		input[i] = round(rand());
	}
	for (i = 0; i < k; i++) {
		kernel[i] = round(rand());
	}

	printf("Start convolution\n");
	clock_t start_time = clock();
	cudaMemcpy(dev_input, input, sizeof(int) * l, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_kernel, kernel, sizeof(int) * k, cudaMemcpyHostToDevice);

	int block = 256;
	int grid = (l + block - 1) / block;
	conv1d<<<grid, block>>>(input, kernel, output, l, k);

	cudaMemcpy(output, dev_output, sizeof(int) * l, cudaMemcpyDeviceToHost);
	clock_t end_time = clock();
	printf("Time consuming of 1D convolution of %d array with %d kernel is %f ms.\n", l, k, static_cast<double>(end_time - start_time)/CLOCKS_PER_SEC*1000);

	cudaFree(dev_input);
	cudaFree(dev_kernel);
	cudaFree(dev_output);
	cudaFreeHost(input);
	cudaFreeHost(kernel);
	cudaFreeHost(output);

	return 0;
}

#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define BLOCK_SIZE 1024

// kernel
__global__ void sumReductionKernel(float* d_input, float* d_output)
{
    __shared__ float output[2 * BLOCK_SIZE];
    
    int startingIndex = 2 * blockIdx.x * blockDim.x;
    
    output[threadIdx.x] = d_input[startingIndex + threadIdx.x];
    output[blockDim.x + threadIdx.x] = d_input[startingIndex + blockDim.x + threadIdx.x];
    
    for(int step = blockDim.x; step > 0; step /= 2)
    {
        __syncthreads();
        
        if(threadIdx.x < step)
        {
            output[threadIdx.x] += output[threadIdx.x + step];
        }
    }
    
    d_output[blockIdx.x] = output[0];
}

// CUDA error checking
void errorCheck(unsigned int line)
{
    cudaError_t cudaError = cudaGetLastError();

    if(cudaError != cudaSuccess)
    {
        printf("CUDA error in line %u in file %s: %s\n", line - 1, __FILE__, cudaGetErrorString(cudaError));
        exit(EXIT_FAILURE);
    }
}

// host function containing kernel call
void sumReduction(float* input, float* output, size_t size)
{
    dim3 numOfBlocks(ceil(size / (2 * (float) BLOCK_SIZE)), 1, 1);
    dim3 numOfThreads(BLOCK_SIZE, 1, 1);

    size_t bytesInput = size * sizeof(float);
    size_t bytesOutput = (size / (2 * BLOCK_SIZE)) * sizeof(float);

    float* d_input;
    float* d_output;

    cudaMalloc((void**) &d_input, bytesInput);
    errorCheck(__LINE__);
    cudaMalloc((void**) &d_output, bytesOutput);
    errorCheck(__LINE__);

    cudaMemcpy(d_input, input, bytesInput, cudaMemcpyHostToDevice);
    errorCheck(__LINE__);

    sumReductionKernel<<<numOfBlocks, numOfThreads>>>(d_input, d_output);
    errorCheck(__LINE__);

    cudaMemcpy(output, d_output, bytesOutput, cudaMemcpyDeviceToHost);
    errorCheck(__LINE__);

    cudaFree(d_input);
    errorCheck(__LINE__);
    cudaFree(d_output);
    errorCheck(__LINE__);
}

int main()
{
    struct timespec start, end;

    srand(time(NULL));

    size_t size = 268435456;

    float* input = (float*) malloc(size * sizeof(float));
    float* output = (float*) malloc((size / (2 * BLOCK_SIZE)) * sizeof(float));
    
    for(int i = 0; i < size; ++i)
    {
        input[i] = rand() % 129 - 64;
    }

    clock_gettime(CLOCK_REALTIME, &start);

    // do sum reduction
    sumReduction(input, output, size);

    clock_gettime(CLOCK_REALTIME, &end);

    time_t execTime = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;

    printf("Execution time: %d microseconds.", execTime);

    return 0;
}

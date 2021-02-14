#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define BLOCK_SIZE 1024

// first kernel - does scan for each block
__global__ void blockSumScanKernel(float* d_input, float* d_output, size_t size)
{
    __shared__ float blockOutput[BLOCK_SIZE];
    
    // indexing variable
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(i < size)
    {
        blockOutput[threadIdx.x] = d_input[i];
        
        for(int step = 1; step <= threadIdx.x; step *= 2)
        {
            __syncthreads();
            
            float chunk = blockOutput[threadIdx.x - step];
            
            __syncthreads();
            
            blockOutput[threadIdx.x] += chunk;
        }
        
        d_output[i] = blockOutput[threadIdx.x];
    }
}

// second kernel - seals together results for all blocks from previous kernel into one output array
__global__ void sealingSumScanKernel(float* d_output, size_t size)
{
    // indexing variable
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    int numOfValues = (i - (i % BLOCK_SIZE)) / BLOCK_SIZE;
    
    for(int j = 1; j <= numOfValues; ++j)
    {
        d_output[i] += d_output[j * BLOCK_SIZE - 1];
    }
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
void sumScan(float* input, float* output, size_t size)
{
    dim3 numOfBlocks(ceil(size / (float) BLOCK_SIZE), 1, 1);
    dim3 numOfThreads(BLOCK_SIZE, 1, 1);

    size_t bytesInput = size * sizeof(float);

    float* d_input;
    float* d_output;

    cudaMalloc((void**) &d_input, bytesInput);
    errorCheck(__LINE__);
    cudaMalloc((void**) &d_output, bytesInput);
    errorCheck(__LINE__);

    cudaMemcpy(d_input, input, bytesInput, cudaMemcpyHostToDevice);
    errorCheck(__LINE__);

    blockSumScanKernel<<<numOfBlocks, numOfThreads>>>(d_input, d_output, size);
    errorCheck(__LINE__);
    
    cudaFree(d_input);
    errorCheck(__LINE__);
    
    sealingSumScanKernel<<<numOfBlocks, numOfThreads>>>(d_output, size);
    errorCheck(__LINE__);

    cudaMemcpy(output, d_output, bytesInput, cudaMemcpyDeviceToHost);
    errorCheck(__LINE__);
    
    cudaFree(d_output);
    errorCheck(__LINE__);
}

int main()
{
    struct timespec start, end;

    srand(time(NULL));

    size_t size = 4194304;

    float* input = (float*) malloc(size * sizeof(float));
    float* output = (float*) malloc(size * sizeof(float));
    
    for(int i = 0; i < size; ++i)
    {
        input[i] = rand() % 129 - 64;
    }
    
    clock_gettime(CLOCK_REALTIME, &start);

    // do sum scan
    sumScan(input, output, size);
    
    clock_gettime(CLOCK_REALTIME, &end);

    time_t execTime = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;

    printf("Execution time: %d microseconds.", execTime);

    return 0;
}

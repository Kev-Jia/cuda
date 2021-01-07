#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define BLOCK_SIZE 1024

// kernel
__global__ void tiledConvolution_1D_Kernel(float* d_m, const float* __restrict__ d_mask, float* d_n, size_t length, size_t maskLength, int N_TILE_LENGTH)
{
    float result = 0;

    // indexing variables
    int n_index = blockIdx.x * N_TILE_LENGTH + threadIdx.x;
    int m_index = n_index - maskLength / 2;

    __shared__ float tile_m[BLOCK_SIZE];

    // thread boundary check for loading input tiles
    if(m_index >= 0 && m_index < length)
    {
        tile_m[threadIdx.x] = d_m[m_index];
    }
    else
    {
        tile_m[threadIdx.x] = 0;
    }
    
    __syncthreads();
    
    // thread boundary check for calculation
    if(threadIdx.x < N_TILE_LENGTH && n_index < length)
    {
        for(int i = 0; i < maskLength; ++i)
        {
            result += d_mask[i] * tile_m[threadIdx.x + i];
        }
        
        // write result
        d_n[n_index] = result;
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
void convolution_1D(float* m, float* mask, float* n, size_t length, size_t maskLength, int N_TILE_LENGTH)
{
    dim3 numOfBlocks(ceil(length / (float) N_TILE_LENGTH), 1, 1);
    dim3 numOfThreads(BLOCK_SIZE, 1, 1);

    size_t bytes_m = length * sizeof(float);
    size_t bytes_mask = maskLength * sizeof(float);

    float* d_m;
    float* d_mask;
    float* d_n;

    cudaMalloc((void**) &d_m, bytes_m);
    errorCheck(__LINE__);
    cudaMalloc((void**) &d_mask, bytes_mask);
    errorCheck(__LINE__);
    cudaMalloc((void**) &d_n, bytes_m);
    errorCheck(__LINE__);

    cudaMemcpy(d_m, m, bytes_m, cudaMemcpyHostToDevice);
    errorCheck(__LINE__);
    cudaMemcpy(d_mask, mask, bytes_mask, cudaMemcpyHostToDevice);
    errorCheck(__LINE__);

    tiledConvolution_1D_Kernel<<<numOfBlocks, numOfThreads>>>(d_m, d_mask, d_n, length, maskLength, N_TILE_LENGTH);
    errorCheck(__LINE__);
    
    cudaMemcpy(n, d_n, bytes_m, cudaMemcpyDeviceToHost);
    errorCheck(__LINE__);
    
    cudaFree(d_m);
    errorCheck(__LINE__);
    cudaFree(d_mask);
    errorCheck(__LINE__);
    cudaFree(d_n);
    errorCheck(__LINE__);
}

int main()
{
    struct timespec start, end;

    clock_gettime(CLOCK_REALTIME, &start);
    
    srand(time(NULL));
    
    size_t length = rand() % 1048577 + 15728640;
    size_t maskLength = 121;
    int N_TILE_LENGTH = BLOCK_SIZE - (maskLength - 1);

    float* m = (float*) malloc(length * sizeof(float));
    float* mask = (float*) malloc(maskLength * sizeof(float));
    float* n = (float*) malloc(length * sizeof(float));

    for(int i = 0; i < length; ++i)
    {
        m[i] = rand() % 129 - 64;
    }
  
    for(int j = 0; j < maskLength; ++j)
    {
        mask[j] = rand() % 1001 / 1000.0;
    }

    // do convolution
    convolution_1D(m, mask, n, length, maskLength, N_TILE_LENGTH);
    
    clock_gettime(CLOCK_REALTIME, &end);

    time_t execTime = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;

    printf("Execution time: %d microseconds.", execTime);

    return 0;
}


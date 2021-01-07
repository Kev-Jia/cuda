#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

// kernel
__global__ void convolution_1D_Kernel(float* d_m, float* d_mask, float* d_n, size_t length, size_t maskLength)
{
    // indexing variables
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int m_index = i - maskLength / 2;
    
    // thread boundary check
    if(i < length)
    {
        for(int j = 0; j < maskLength; ++j)
        {
            if(m_index + j >= 0 && m_index + j < length)
            {
                d_n[i] += d_m[m_index + j] * d_mask[j];
            }
        }
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
void convolution_1D(float* m, float* mask, float* n, size_t length, size_t maskLength)
{
    dim3 numOfBlocks(ceil(length / 1024.0), 1, 1);
    dim3 numOfThreads(1024, 1, 1);

    size_t bytes_m = length * sizeof(float);
    size_t bytes_mask = maskLength * sizeof(float);
    size_t bytes_n = length * sizeof(float);

    float* d_m;
    float* d_mask;
    float* d_n;

    cudaMalloc((void**) &d_m, bytes_m);
    errorCheck(__LINE__);
    cudaMalloc((void**) &d_mask, bytes_mask);
    errorCheck(__LINE__);
    cudaMalloc((void**) &d_n, bytes_n);
    errorCheck(__LINE__);

    cudaMemcpy(d_m, m, bytes_m, cudaMemcpyHostToDevice);
    errorCheck(__LINE__);
    cudaMemcpy(d_mask, mask, bytes_mask, cudaMemcpyHostToDevice);
    errorCheck(__LINE__);

    convolution_1D_Kernel<<<numOfBlocks, numOfThreads>>>(d_m, d_mask, d_n, length, maskLength);
    errorCheck(__LINE__);
    
    cudaMemcpy(n, d_n, bytes_n, cudaMemcpyDeviceToHost);
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
    convolution_1D(m, mask, n, length, maskLength);
    
    clock_gettime(CLOCK_REALTIME, &end);

    time_t execTime = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;

    printf("Execution time: %d microseconds.", execTime);

    return 0;
}

#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

// kernel
__global__ void convolution_2D_Kernel(float* d_m, float* d_mask, float* d_n, size_t a, size_t b, size_t maskWidth)
{
    // indexing variables
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int m_row = j - maskWidth / 2;
    int m_col = i - maskWidth / 2;

    // thread boundary check
    if(i < b && j < a)
    {
        for(int k = 0; k < maskWidth; ++k)
        {
            for(int l = 0; l < maskWidth; ++l)
            {
                if(m_row + l >= 0 && m_row + l < a && m_col + k >= 0 && m_col + k < b)
                {
                    d_n[j * b + i] += d_m[(m_row + l) * b + m_col + k] * d_mask[l * maskWidth + k];
                }
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
void convolution_2D(float* m, float* mask, float* n, size_t a, size_t b, size_t maskWidth)
{
    dim3 numOfBlocks(ceil(b / 32.0), ceil(a / 32.0), 1);
    dim3 numOfThreads(32, 32, 1);
    
    size_t bytes_m = a * b * sizeof(float);
    size_t bytes_mask = maskWidth * maskWidth * sizeof(float);
    size_t bytes_n = a * b * sizeof(float);

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

    convolution_2D_Kernel<<<numOfBlocks, numOfThreads>>>(d_m, d_mask, d_n, a, b, maskWidth);
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

    srand(time(NULL));

    size_t a = rand() % 257 + 3840;
    size_t b = rand() % 257 + 3840;
    size_t maskWidth = 11;

    float* m = (float*) malloc(a * b * sizeof(float));
    float* mask = (float*) malloc(maskWidth * maskWidth * sizeof(float));
    float* n = (float*) malloc(a * b * sizeof(float));

    for(int i = 0; i < a * b; ++i)
    {
        m[i] = rand() % 129 - 64;
    }

    for(int j = 0; j < maskWidth * maskWidth; ++j)
    {
        mask[j] = rand() % 1001 / 1000.0;
    }

    clock_gettime(CLOCK_REALTIME, &start);
    
    // do convolution
    convolution_2D(m, mask, n, a, b, maskWidth);

    clock_gettime(CLOCK_REALTIME, &end);

    time_t execTime = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;

    printf("Execution time: %d microseconds.", execTime);

    return 0;
}

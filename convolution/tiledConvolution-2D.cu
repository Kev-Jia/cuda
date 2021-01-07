#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define BLOCK_WIDTH 32

// kernel
__global__ void tiledConvolution_2D_Kernel(float* d_m, const float* __restrict__ d_mask, float* d_n, size_t a, size_t b, size_t maskWidth, int N_TILE_WIDTH)
{
    float result = 0;
   
    // indexing variables
    int n_row = blockIdx.y * N_TILE_WIDTH + threadIdx.y;
    int n_col = blockIdx.x * N_TILE_WIDTH + threadIdx.x;
    
    int m_row = n_row - maskWidth / 2;
    int m_col = n_col - maskWidth / 2;
    
    __shared__ float tile_m[BLOCK_WIDTH][BLOCK_WIDTH];
    
    // thread boundary check for loading input tiles
    if(m_row >= 0 && m_row < a && m_col >= 0 && m_col < b)
    {
        tile_m[threadIdx.y][threadIdx.x] = d_m[m_row * b + m_col];
    }
    else
    {
        tile_m[threadIdx.y][threadIdx.x] = 0;
    }
    
    __syncthreads();
    
    // thread boundary check for calculation
    if(threadIdx.y < N_TILE_WIDTH && threadIdx.x < N_TILE_WIDTH && n_row < a && n_col < b)
    {
        for(int i = 0; i < maskWidth; ++i)
        {
            for(int j = 0; j < maskWidth; ++j)
            {
                result += d_mask[i * maskWidth + j] * tile_m[threadIdx.y + i][threadIdx.x + j];
            }
        }
        
        // write result 
        d_n[n_row * b + n_col] = result;
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
void convolution_2D(float* m, float* mask, float* n, size_t a, size_t b, size_t maskWidth, int N_TILE_WIDTH)
{
    dim3 numOfBlocks(ceil(b / (float) N_TILE_WIDTH), ceil(a / (float) N_TILE_WIDTH), 1);
    dim3 numOfThreads(BLOCK_WIDTH, BLOCK_WIDTH, 1);

    size_t bytes_m = a * b * sizeof(float);
    size_t bytes_mask = maskWidth * maskWidth * sizeof(float);

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

    tiledConvolution_2D_Kernel<<<numOfBlocks, numOfThreads>>>(d_m, d_mask, d_n, a, b, maskWidth,  N_TILE_WIDTH);
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
    
    size_t a = rand() % 257 + 3840;
    size_t b = rand() % 257 + 3840;
    size_t maskWidth = 11;
    
    int N_TILE_WIDTH = BLOCK_WIDTH - (maskWidth - 1);

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

    // do convolution
    convolution_2D(m, mask, n, a, b, maskWidth, N_TILE_WIDTH);
    
    clock_gettime(CLOCK_REALTIME, &end);

    time_t execTime = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;

    printf("Execution time: %d microseconds.", execTime);

    return 0;
}

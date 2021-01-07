#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define TILE_WIDTH 32

// kernel
__global__ void tiledMultiplyMatricesKernel(float* d_x, float* d_y, float* d_z, int m, int n, int p)
{
    __shared__ float tile_x[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_y[TILE_WIDTH][TILE_WIDTH];
    
    // indexing variables
    int rowNum = blockIdx.y * blockDim.y + threadIdx.y;
    int colNum = blockIdx.x * blockDim.x + threadIdx.x;

    float result = 0;

    for(int i = 0; i < ceil(n / (float) TILE_WIDTH); ++i)
    {
        // thread boundary checks for loading input tiles
        if(rowNum < m && (i * TILE_WIDTH + threadIdx.x) < n)
        {
            tile_x[threadIdx.y][threadIdx.x] = d_x[rowNum * n + i * TILE_WIDTH + threadIdx.x];
        }
        else
        {
            tile_x[threadIdx.y][threadIdx.x] = 0.0;    
        }
        if((i * TILE_WIDTH + threadIdx.y) < n && colNum < p)
        {
            tile_y[threadIdx.y][threadIdx.x] = d_y[(i * TILE_WIDTH + threadIdx.y) * p + colNum];
        }
        else
        {
            tile_x[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();
        
        for(int j = 0; j < TILE_WIDTH; ++j)
        {
            result += tile_x[threadIdx.y][j] * tile_y[j][threadIdx.x];
        }

        __syncthreads();
    }

    // thread boundary check for writing output
    if(rowNum < m && colNum < p)
    {
        d_z[rowNum * p + colNum] = result;
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
void multiplyMatrices(float* x, float* y, float* z, int m, int n, int p)
{
    dim3 numOfBlocks(ceil(p / (float) TILE_WIDTH), ceil(m / (float) TILE_WIDTH), 1);
    dim3 numOfThreads(TILE_WIDTH, TILE_WIDTH, 1);

    size_t bytes_x = m * n * sizeof(float);
    size_t bytes_y = n * p * sizeof(float);
    size_t bytes_z = m * p * sizeof(float);

    float* d_x;
    float* d_y;
    float* d_z;

    cudaMalloc((void**) &d_x, bytes_x);
    errorCheck(__LINE__);
    cudaMalloc((void**) &d_y, bytes_y);
    errorCheck(__LINE__);
    cudaMalloc((void**) &d_z, bytes_z);
    errorCheck(__LINE__);

    cudaMemcpy(d_x, x, bytes_x, cudaMemcpyHostToDevice);
    errorCheck(__LINE__);
    cudaMemcpy(d_y, y, bytes_y, cudaMemcpyHostToDevice);
    errorCheck(__LINE__);

    tiledMultiplyMatricesKernel<<<numOfBlocks, numOfThreads>>>(d_x, d_y, d_z, m, n, p);
    errorCheck(__LINE__);

    cudaMemcpy(z, d_z, bytes_z, cudaMemcpyDeviceToHost);
    errorCheck(__LINE__);

    cudaFree(d_x);
    errorCheck(__LINE__);
    cudaFree(d_y);
    errorCheck(__LINE__);
    cudaFree(d_z);
    errorCheck(__LINE__);
}

int main()
{
    struct timespec start, end;
    
    clock_gettime(CLOCK_REALTIME, &start);

    srand(time(NULL));

    size_t m = rand() % 257 + 3840;
    size_t n = rand() % 257 + 3840;
    size_t p = rand() % 257 + 3840;

    float* x = (float*) malloc(m * n * sizeof(float));
    float* y = (float*) malloc(n * p * sizeof(float));
    float* z = (float*) malloc(m * p * sizeof(float));

    for(int i = 0; i < m * n; ++i)
    {
        x[i] = rand() % 129 - 64;
    }

    for(int i = 0; i < n * p; ++i)
    {
        y[i] = rand() % 129 - 64;
    }

    // do matrix multiplication
    multiplyMatrices(x, y, z, m, n, p);

    clock_gettime(CLOCK_REALTIME, &end);

    time_t execTime = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;

    printf("Execution time: %d microseconds.", execTime);

    return 0;
}



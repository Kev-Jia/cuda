#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define TILE_WIDTH 16

__global__ void tiledMultiplyMatricesKernel(float* d_x, float* d_y, float* d_z, int m, int n, int p)
{
    __shared__ float tile_x[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_y[TILE_WIDTH][TILE_WIDTH];
    
    int T_x = threadIdx.x;
    int T_y = threadIdx.y;
    
    int B_x = blockIdx.x;
    int B_y = blockIdx.y;

    int rowNum = B_y * blockDim.y + T_y;
    int colNum = B_x * blockDim.x + T_x;

    float result = 0;

    for(int i = 0; i < n / TILE_WIDTH; ++i)
    {
        tile_x[T_y][T_x] = d_x[rowNum * n + i * TILE_WIDTH + T_x];
        tile_y[T_y][T_x] = d_y[(i * TILE_WIDTH + T_y) * p + colNum];
        __syncthreads();

        for(int j = 0; j < TILE_WIDTH; ++j)
        {
            result += tile_x[T_y][j] * tile_y[j][T_x];
        }
        __syncthreads();
    }
    d_z[rowNum * p + colNum] = result;
}

void errorCheck(unsigned int line)
{
    cudaError_t cudaError = cudaGetLastError();
    if(cudaError != cudaSuccess)
    {
        printf("CUDA error in line %u in file %s: %s\n", line - 1, __FILE__, cudaGetErrorString(cudaError));
        exit(EXIT_FAILURE);
    }
}

void multiplyMatrices(float* x, float* y, float* z, int m, int n, int p)
{
    dim3 numOfBlocks(ceil(p / (float) TILE_WIDTH), ceil(m / (float) TILE_WIDTH), 1);
    dim3 numOfThreads(TILE_WIDTH, TILE_WIDTH, 1);

    size_t elements_x = m * n * sizeof(float);
    size_t elements_y = n * p * sizeof(float);
    size_t elements_z = m * p * sizeof(float);

    float* d_x;
    float* d_y;
    float* d_z;

    cudaMalloc((void**) &d_x, elements_x);
    errorCheck(__LINE__);
    cudaMalloc((void**) &d_y, elements_y);
    errorCheck(__LINE__);
    cudaMalloc((void**) &d_z, elements_z);
    errorCheck(__LINE__);

    cudaMemcpy(d_x, x, elements_x, cudaMemcpyHostToDevice);
    errorCheck(__LINE__);
    cudaMemcpy(d_y, y, elements_y, cudaMemcpyHostToDevice);
    errorCheck(__LINE__);

    tiledMultiplyMatricesKernel<<<numOfBlocks, numOfThreads>>>(d_x, d_y, d_z, m, n, p);
    errorCheck(__LINE__);

    cudaMemcpy(z, d_z, elements_z, cudaMemcpyDeviceToHost);
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
    size_t m = 4096;
    size_t n = 4096;
    size_t p = 4096;

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

    for(int i = 0; i < m * p; ++i)
    {
        z[i] = 0;
    }

    multiplyMatrices(x, y, z, m, n, p);

    clock_gettime(CLOCK_REALTIME, &end);
    time_t execTime = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec + start.tv_nsec) / 1000;

    printf("Execution time: %d microseconds.", execTime);

    return 0;
}



#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

// kernel
__global__ void multiplyMatricesKernel(float* d_x, float* d_y, float* d_z, int m, int n, int p)
{
    // indexing variable
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    // thread boundary check
    if(i < m)
    {
        for(int j = 0; j < p; ++j)
        {
            for(int k = 0; k < n; ++k)
            {
                d_z[i * p + j] += d_x[i * n + k] * d_y[k * p + j];
            }
        }
    }
}

// host function containing kernel call
void multiplyMatrices(float* x, float* y, float* z, int m, int n, int p)
{
    dim3 numOfBlocks(ceil(m / 1024.0), 1, 1);
    dim3 numOfThreads(1024, 1, 1);

    int bytes_x = m * n * sizeof(float);
    int bytes_y = n * p * sizeof(float);
    int bytes_z = m * p * sizeof(float);

    float* d_x;
    float* d_y;
    float* d_z;

    cudaMalloc((void**) &d_x, bytes_x);
    cudaMalloc((void**) &d_y, bytes_y);
    cudaMalloc((void**) &d_z, bytes_z);

    cudaMemcpy(d_x, x, bytes_x, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_y, y, bytes_y, cudaMemcpyHostToDevice);

    multiplyMatricesKernel<<<numOfBlocks, numOfThreads>>>(d_x, d_y, d_z, m, n, p);

    cudaMemcpy(z, d_z, bytes_z, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
}

int main()
{
    struct timespec start, end;

    clock_gettime(CLOCK_REALTIME, &start);

    srand(time(NULL));

    size_t m = rand() % 193 + 1856;
    size_t n = rand() % 193 + 1856;
    size_t p = rand() % 193 + 1856;

    float* x = (float*) malloc(m * n * sizeof(float));
    float* y = (float*) malloc(n * p * sizeof(float));
    float* z = (float*) malloc(m * p * sizeof(float));

    for(int i = 0; i < sizeof(x) / sizeof(float); ++i)
    {
        x[i] = rand() % 129 - 64;
    }

    for(int i = 0; i < sizeof(y) / sizeof(float); ++i)
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

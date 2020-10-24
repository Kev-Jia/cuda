#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <stdlib.h>

__global__ void multiplyMatricesKernel(float* d_x, float* d_y, float* d_z, int m, int n, int p)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if(i < p && j < m)
    {
        for(int k = 0; k < n; ++k)
        {
            d_z[j * p + i] += d_x[j * n + k] * d_y[k * p + i];
        }
    }
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
    dim3 numOfBlocks(ceil(m / 32.0), ceil(p / 32.0), 1);
    dim3 numOfThreads(32, 32, 1);

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

    multiplyMatricesKernel<<<numOfBlocks, numOfThreads>>>(d_x, d_y, d_z, m, n, p);
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
    srand(time(NULL));

    size_t m = rand() % 8 + 1;
    size_t n = rand() % 8 + 1;
    size_t p = rand() % 8 + 1;

    float x[m * n] = {0};
    float y[n * p] = {0};
    float z[m * p] = {0};

    printf("X =\n[");
    for(int i = 0; i < sizeof(x) / sizeof(float); ++i)
    {
        x[i] = rand() % 129 - 64;
        printf("%.1f ", x[i]);
        if((i + 1) % n == 0 && i != (sizeof(x) / sizeof(float) - 1))
        {
            printf("]\n[");
        }
        if(i == (sizeof(x) / sizeof(float) - 1))
        {
            printf("]\n\n");
        }
    }
    
    printf("Y = \n[");
    for(int i = 0; i < sizeof(y) / sizeof(float); ++i)
    {
        y[i] = rand() % 129 - 64;
        printf("%.1f ", y[i]);
        if((i + 1) % p == 0 && i != (sizeof(y) / sizeof(float) - 1))
        {
            printf("]\n[");
        }
        if(i == (sizeof(y) / sizeof(float) - 1))
        {
            printf("]\n\n");
        }
    }

    multiplyMatrices(x, y, z, m, n, p);

    printf("Z = \n[");
    for(int i = 0; i < sizeof(z) / sizeof(float); ++i)
    {   
        printf("%.1f ", z[i]);
        if((i + 1) % p == 0 && i != (sizeof(z) / sizeof(float) - 1))
        {
            printf("]\n[");
        }
        if(i == (sizeof(z) / sizeof(float) - 1))
        {
            printf("]\n\n");
        }
    }
    return 0;
}

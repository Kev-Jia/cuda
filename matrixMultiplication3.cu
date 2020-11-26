#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <stdlib.h>

__global__ void multiplyMatricesKernel(float* d_x, float* d_y, float* d_z, int m, int n, int p)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
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

void multiplyMatrices(float* x, float* y, float* z, int m, int n, int p)
{
    int elements_x = m * n * sizeof(float);
    int elements_y = n * p * sizeof(float);
    int elements_z = m * p * sizeof(float);

    float* d_x;
    float* d_y;
    float* d_z;

    cudaMalloc((void**) &d_x, elements_x);
    cudaMalloc((void**) &d_y, elements_y);
    cudaMalloc((void**) &d_z, elements_z);

    cudaMemcpy(d_x, x, elements_x, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_y, y, elements_y, cudaMemcpyHostToDevice);

    multiplyMatricesKernel<<<ceil(m / 64.0), 64>>>(d_x, d_y, d_z, m, n, p);

    cudaMemcpy(z, d_z, elements_z, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
}

int main()
{
    srand(time(NULL));
    int m = rand() % 8 + 1;
    int n = rand() % 8 + 1;
    int p = rand() % 8 + 1;

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

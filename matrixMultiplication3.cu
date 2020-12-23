// include necessary libs
#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

// CUDA kernel function
__global__ void multiplyMatricesKernel(float* d_x, float* d_y, float* d_z, int m, int n, int p)
{
    // define and initialize the variable that will be used in indexing - this is for brevity
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    // only have threads within the range of the amount of rows of the result matrix to calculate an element of it
    // this is to prevent global memory segfaults
    if(i < m)
    {
        // calculate each row of the result matrix
        for(int j = 0; j < p; ++j)
        {
            for(int k = 0; k < n; ++k)
            {
                d_z[i * p + j] += d_x[i * n + k] * d_y[k * p + j];
            }
        }
    }
}

// host function that calls CUDA kernel
void multiplyMatrices(float* x, float* y, float* z, int m, int n, int p)
{
    // define and initialize dimension variables containing data regarding the dimensions of the grid and dimensions of each block
    dim3 numOfBlocks(ceil(m / 64.0), 1, 1);
    dim3 numOfThreads(64, 1, 1);

    // define and initialize the variables containing number of bytes in each matrix 
    int bytes_x = m * n * sizeof(float);
    int bytes_y = n * p * sizeof(float);
    int bytes_z = m * p * sizeof(float);

    // define the pointers that will point to the start of allocated device memory for each matrix
    float* d_x;
    float* d_y;
    float* d_z;

    // allocate global memory for the matrices on the device
    cudaMalloc((void**) &d_x, bytes_x);
    cudaMalloc((void**) &d_y, bytes_y);
    cudaMalloc((void**) &d_z, bytes_z);

    // copy the data of input matrices to the allocated global memory on the device
    cudaMemcpy(d_x, x, bytes_x, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_y, y, bytes_y, cudaMemcpyHostToDevice);

    // call the CUDA kernel
    multiplyMatricesKernel<<<ceil(m / 64.0), 64>>>(d_x, d_y, d_z, m, n, p);
    
    // copy the data of the result matrix from global memory to host DRAM
    cudaMemcpy(z, d_z, bytes_z, cudaMemcpyDeviceToHost);
    
    // free the allocated device global memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
}

int main()
{
    // initialize pseudo-random number generator with seed of the current seconds since 01/01/1970
    srand(time(NULL));
    
    // define and initialize dimension variables to be random values from 1 to 8 for the 3 matrices
    int m = rand() % 8 + 1;
    int n = rand() % 8 + 1;
    int p = rand() % 8 + 1;

    // statically define all the matrices
    float x[m * n];
    float y[n * p];
    float z[m * p];

    // output matrix x and beforehand set each of its elements to a pseudo-random value from -64 to 64
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

    // output matrix y and beforehand set each of its elements to a pseudo-random value from -64 to 64
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
    
    // multiply the input matrices x and y
    multiplyMatrices(x, y, z, m, n, p);
    
    // output result matrix z
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
    
    // exit program
    return 0;
}

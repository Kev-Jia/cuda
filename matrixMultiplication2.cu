// include necessary libs
#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

// CUDA kernel function
__global__ void multiplyMatricesKernel(float* d_x, float* d_y, float* d_z, int m, int n, int p)
{
    // define and initialize variables that will be used in indexing - this is for brevity
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    
    // only have threads within the range of the result matrix dimensions calculate an element of it
    // this is to prevent global memory segfaults
    if(i < p && j < m)
    {
        for(int k = 0; k < n; ++k)
        {   
            // calculate the dot product of each element of the result matrix
            d_z[j * p + i] += d_x[j * n + k] * d_y[k * p + i];
        }
    }
}

// error checking function - checks for CUDA errors
void errorCheck(unsigned int line)
{
    // get most recent CUDA error
    cudaError_t cudaError = cudaGetLastError();
    
    // if error code wasn't a code describing success
    if(cudaError != cudaSuccess)
    {
        // output that there has been a CUDA error in the line of the CUDA function call
        // and exit the program
        printf("CUDA error in line %u in file %s: %s\n", line - 1, __FILE__, cudaGetErrorString(cudaError));
        exit(EXIT_FAILURE);
    }
}

// host function that calls CUDA kernel
void multiplyMatrices(float* x, float* y, float* z, int m, int n, int p)
{
    // define and initialize dimension variables containing data regarding the dimensions of the grid and dimensions of each block
    dim3 numOfBlocks(ceil(p / 32.0), ceil(m / 32.0), 1);
    dim3 numOfThreads(32, 32, 1);
    
    // define and initialize the variables containing number of bytes in each matrix
    size_t elements_x = m * n * sizeof(float);
    size_t elements_y = n * p * sizeof(float);
    size_t elements_z = m * p * sizeof(float);
    
    // define the pointers that will point to the start of allocated device memory for each matrix
    float* d_x;
    float* d_y;
    float* d_z;
    
    // allocate global memory for the matrices on the device and check for CUDA errors
    cudaMalloc((void**) &d_x, elements_x);
    errorCheck(__LINE__);
    cudaMalloc((void**) &d_y, elements_y);
    errorCheck(__LINE__);
    cudaMalloc((void**) &d_z, elements_z);
    errorCheck(__LINE__);
    
    // copy the data of input matrices to the allocated global memory on the device and check for CUDA errors
    cudaMemcpy(d_x, x, elements_x, cudaMemcpyHostToDevice);
    errorCheck(__LINE__);
    cudaMemcpy(d_y, y, elements_y, cudaMemcpyHostToDevice);
    errorCheck(__LINE__);

    // call the kernel and check for CUDA errors
    multiplyMatricesKernel<<<numOfBlocks, numOfThreads>>>(d_x, d_y, d_z, m, n, p);
    errorCheck(__LINE__);

    // copy the data of the result matrix from device global memory to host DRAM and check for CUDA errors
    cudaMemcpy(z, d_z, elements_z, cudaMemcpyDeviceToHost);
    errorCheck(__LINE__);

    // free the allocated device global memory and check for CUDA errors
    cudaFree(d_x);
    errorCheck(__LINE__);
    cudaFree(d_y);
    errorCheck(__LINE__);
    cudaFree(d_z);
    errorCheck(__LINE__);
}

int main()
{
    // initialize pseudo-random number generator with seed of the current seconds since 01/01/1970
    srand(time(NULL));

    // define and initialize dimension variables to be random values from 1 to 8 for the 3 matrices
    size_t m = rand() % 8 + 1;
    size_t n = rand() % 8 + 1;
    size_t p = rand() % 8 + 1;

    // statically define and initialize the matrices to be entirely 0
    float x[m * n] = {0};
    float y[n * p] = {0};
    float z[m * p] = {0};

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

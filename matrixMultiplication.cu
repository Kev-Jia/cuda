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
        // calculate the dot product of each element of the result matrix
        for(int k = 0; k < n; ++k)
        {
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
    size_t bytes_x = m * n * sizeof(float);
    size_t bytes_y = n * p * sizeof(float);
    size_t bytes_z = m * p * sizeof(float);
    
    // define the pointers that will point to the start of allocated device memory for each matrix
    float* d_x;
    float* d_y;
    float* d_z;
    
    // allocate global memory for the matrices on the device and check for CUDA errors
    cudaMalloc((void**) &d_x, bytes_x);
    errorCheck(__LINE__);
    cudaMalloc((void**) &d_y, bytes_y);
    errorCheck(__LINE__);
    cudaMalloc((void**) &d_z, bytes_z);
    errorCheck(__LINE__);
    
    // copy the data of input matrices to the allocated global memory on the device and check for CUDA errors
    cudaMemcpy(d_x, x, bytes_x, cudaMemcpyHostToDevice);
    errorCheck(__LINE__);
    cudaMemcpy(d_y, y, bytes_y, cudaMemcpyHostToDevice);
    errorCheck(__LINE__);

    // call the CUDA kernel and check for CUDA errors
    multiplyMatricesKernel<<<numOfBlocks, numOfThreads>>>(d_x, d_y, d_z, m, n, p);
    errorCheck(__LINE__);

    // copy the data of the result matrix from global memory to host DRAM and check for CUDA errors
    cudaMemcpy(z, d_z, bytes_z, cudaMemcpyDeviceToHost);
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
    // define structs that will enable us to get exec time of the program
    struct timespec start, end;
    
    // get the details regarding start time of this program and store it in the start struct
    clock_gettime(CLOCK_REALTIME, &start);

    // initialize pseudo-random number generator with seed of the current seconds since 01/01/1970
    srand(time(NULL));
    
    // define and initialize dimension variables for the 3 matrices
    size_t m = rand() % 257 + 3840;
    size_t n = rand() % 257 + 3840;
    size_t p = rand() % 257 + 3840;

    // dynamically allocate DRAM memory for the matrices to account for the matrices being perhaps too big to be statically allocated
    float* x = (float*) malloc(m * n * sizeof(float));
    float* y = (float*) malloc(n * p * sizeof(float));
    float* z = (float*) malloc(m * p * sizeof(float));
    
    // assign a pseudo-random value from -64 to 64 for each element in input matrix x
    for(int i = 0; i < m * n; ++i)
    {
        x[i] = rand() % 129 - 64;
    }
    
    // assign a pseudo-random value from -64 to 64 for each element in input matrix y
    for(int i = 0; i < n * p; ++i)
    {
        y[i] = rand() % 129 - 64;
    }
    
    // multiply the input matrices x and y
    multiplyMatrices(x, y, z, m, n, p);
    
    // get the details regarding the end time of this program and store it in the end struct
    clock_gettime(CLOCK_REALTIME, &end);
    
    // calculate exec time in microseconds
    time_t execTime = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
    
    // output exec time
    printf("Execution time: %d microseconds.", execTime);

    // exit program
    return 0;
}

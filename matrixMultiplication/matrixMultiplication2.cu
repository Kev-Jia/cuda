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
    dim3 numOfBlocks(ceil(m / 1024.0), 1, 1);
    dim3 numOfThreads(1024, 1, 1);

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
    multiplyMatricesKernel<<<numOfBlocks, numOfThreads>>>(d_x, d_y, d_z, m, n, p);
    
    // copy the data of the result matrix from global memory to host DRAM
    cudaMemcpy(z, d_z, bytes_z, cudaMemcpyDeviceToHost);
    
    // free the allocated global memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
}

int main()
{
    // define structs that will enable us to get the exec time of the program
    struct timespec start, end;

    // get the details regarding the start time of this program and store it in the start struct
    clock_gettime(CLOCK_REALTIME, &start);

    // initialize pseudo-random number generator with seed of the current seconds since 01/01/1970
    srand(time(NULL));
    
    // define and initialize dimension variables for the 3 matrices
    // these variables will have values in a range from 1856 to 2048
    size_t m = rand() % 193 + 1856;
    size_t n = rand() % 193 + 1856;
    size_t p = rand() % 193 + 1856;

    // dynamically allocate DRAM memory for the matrices to account for them being pehaps too big to be statically allocated
    float* x = (float*) malloc(m * n * sizeof(float));
    float* y = (float*) malloc(n * p * sizeof(float));
    float* z = (float*) malloc(m * p * sizeof(float));

    // assign a pseudo-random integer value from -64 to 64 for each element in input matrix x
    for(int i = 0; i < sizeof(x) / sizeof(float); ++i)
    {
        x[i] = rand() % 129 - 64;
    }
        
    // assign a pseudo-random integer value from -64 to 64 for each element in input matrix y
    for(int i = 0; i < sizeof(y) / sizeof(float); ++i)
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

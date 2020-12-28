// include necessary libs
#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

// define constant TILE_WIDTH
#define TILE_WIDTH 32

// CUDA kernel function
__global__ void tiledMultiplyMatricesKernel(float* d_x, float* d_y, float* d_z, int m, int n, int p)
{
    // define statically allocated 2D-indexed shared memory tile matrices
    __shared__ float tile_x[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_y[TILE_WIDTH][TILE_WIDTH];
    
    // define and initialize the variables that will be used for indexing - this is for brevity
    int rowNum = blockIdx.y * blockDim.y + threadIdx.y;
    int colNum = blockIdx.x * blockDim.x + threadIdx.x;

    // this variable will prevent writing conflicts to result matrix in global memory
    // also reduces global memory traffic
    // as automatic variables are stored in register memory
    float result = 0;

    // each block will generate one tile of the result matrix
    // have each thread iterate through tiles of the matrices row-wise
    // each thread will end up generating one element of result matrix result iteratively using tiles
    // each iteration will have a partially generate a tile of the result matrix
    for(int i = 0; i < n / TILE_WIDTH; ++i)
    {
        // load elements of d_x and d_y into their respective positions in their tiles
        tile_x[threadIdx.y][threadIdx.x] = d_x[rowNum * n + i * TILE_WIDTH + threadIdx.x];
        tile_y[threadIdx.y][threadIdx.x] = d_y[(i * TILE_WIDTH + threadIdx.y) * p + colNum];

        // sync all the threads in the block so faster threads don't work with uninitialized memory
        __syncthreads();
        
        // calculate a part of the dot product of each element of the result matrix
        for(int j = 0; j < TILE_WIDTH; ++j)
        {
            result += tile_x[threadIdx.y][j] * tile_y[j][threadIdx.x];
        }

        // sync all the threads to prevent the contents of the shared memory being overwritten by faster threads when they finish one iteration of the most outside for loop
        __syncthreads();
    }

    // write results to respectively positioned elements of result
    d_z[rowNum * p + colNum] = result;
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

// host function that calls the CUDA kernel
void multiplyMatrices(float* x, float* y, float* z, int m, int n, int p)
{
    // define and initialize dimension variables containing data regarding the dimensions of the grid and dimensions of each block
    dim3 numOfBlocks(ceil(p / (float) TILE_WIDTH), ceil(m / (float) TILE_WIDTH), 1);
    dim3 numOfThreads(TILE_WIDTH, TILE_WIDTH, 1);

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
    tiledMultiplyMatricesKernel<<<numOfBlocks, numOfThreads>>>(d_x, d_y, d_z, m, n, p);
    errorCheck(__LINE__);

    // copy the data of the result matrix from global memory to host DRAM and check for CUDA errors
    cudaMemcpy(z, d_z, bytes_z, cudaMemcpyDeviceToHost);
    errorCheck(__LINE__);

    // free the allocated global memory and check for CUDA errors
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

    // get the details regarding the start time of this program and store it in the start struct
    clock_gettime(CLOCK_REALTIME, &start);

    // initialize pseudo-random number generator with seed of the current seconds since 01/01/1970
    srand(time(NULL));

    // define and initialize dimension variables for the 3 matrices
    size_t m = 4096;
    size_t n = 4096;
    size_t p = 4096;

    // dynamically allocate DRAM memory for the matrices to account for them being perhaps too big to be statically allocated
    float* x = (float*) malloc(m * n * sizeof(float));
    float* y = (float*) malloc(n * p * sizeof(float));
    float* z = (float*) malloc(m * p * sizeof(float));

    // assign a pseudo-random integer value from -64 to 64 for each element in input matrix x
    for(int i = 0; i < m * n; ++i)
    {
        x[i] = rand() % 129 - 64;
    }

    // assign a pseudo-random integer value from -64 to 64 for each element in input matrix y
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



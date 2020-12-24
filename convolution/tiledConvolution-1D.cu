// include necessary libs
#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define BLOCK_WIDTH 1024


// initialize pseudo-random number generator with seed of current seconds since 01/01/1970
srand(time(NULL));

// define and initialize mask array size variable as global
// this is needed as maskLength is needed to define and initialize global variable TILE_WIDTH
size_t maskLength = 2 * (rand() % 2049 + 30720) - 1;

// define and initialize tile size variable
size_t TILE_WIDTH = BLOCK_WIDTH - (maskLength - 1);

__global__ void tiledConvolution_1D_Kernel(float* d_m, float* d_mask, float* d_n, size_t length, size_t maskLength)
{

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
void convolution_1D(float* m, float* mask, float* n, size_t length, size_t maskLength)
{
    // define and initialize dimension variables containing data regarding the dimensions of the grid and the dimensions of each block
    dim3 numOfBlocks(ceil(length / BLOCK_SIZE), 1, 1);
    dim3 numOfThreads(BLOCK_SIZE, 1, 1);

    // define and initialize variables containing the number of bytes in each array
    size_t bytes_m = length * sizeof(float);
    size_t bytes_mask = maskLength * sizeof(float);
    size_t bytes_n = length * sizeof(float);

    // define the pointers that will point to the start of allocated device memory for each array
    float* d_m;
    float* d_mask;
    float* d_n;

    // allocate global memory for each array on the device and check for CUDA errors
    cudaMalloc((void**) &d_m, bytes_m);
    errorCheck(__LINE__);
    cudaMalloc((void**) &d_mask, bytes_mask);
    errorCheck(__LINE__);
    cudaMalloc((void**) &d_n, bytes_n);
    errorCheck(__LINE__);

    // copy the data of each array to allocated global memory on the device and check for CUDA errors
    cudaMemcpy(d_m, m, bytes_m, cudaMemcpyHostToDevice);
    errorCheck(__LINE__);
    cudaMemcpy(d_mask, mask, bytes_mask, cudaMemcpyHostToDevice);
    errorCheck(__LINE__);

    // call the CUDA kernel and check for CUDA errors
    tiledConvolution_1D_Kernel<<<numOfBlocks, numOfThreads>>>(d_m, d_mask, d_n, length, maskLength);
    errorCheck(__LINE__);
    
    // copy the data of the result array from global memory to host DRAM and check for CUDA errors
    cudaMemcpy(n, d_n, bytes_n, cudaMemcpyDeviceToHost);
    errorCheck(__LINE__);
    
    // free the allocated global memory and check for CUDA errors
    cudaFree(d_m);
    errorCheck(__LINE__);
    cudaFree(d_mask);
    errorCheck(__LINE__);
    cudaFree(d_n);
    errorCheck(__LINE__);
}

int main()
{
    // define structs that will enable us to get the exec time of the program
    struct timespec start, end;

    // get the details regarding the start time of this program and store it in the start struct
    clock_gettime(CLOCK_REALTIME, &start);
    
    // define and initialize size variables for each array
    // the input and result arrays have the same size and thus share a size variable
    size_t length = rand() % 65537 + 983040;
    size_t maskLength = 2 * (rand() % 2049 + 30720) - 1;

    // dynamically allocate DRAM memory for the arrays to account for them perhaps being too big to be statically allocated
    float* m = (float*) malloc(length * sizeof(float));
    float* mask = (float*) malloc(maskLength * sizeof(float));
    float* n = (float*) malloc(length * sizeof(float));

    // assign a pseudo-random integer value from -64 to 64 for each element in input array m
    for(int i = 0; i < length; ++i)
    {
        m[i] = rand() % 129 - 64;
    }
  
    // assign a pseudo-random float value from 0 to 1 with a precision of 3 decimal places for each element in mask array
    for(int j = 0; j < maskLength; ++j)
    {
       mask[j] =  rand() % 1001 / 1000.0;
    }

    // perform 1D convolution operation on input array m using a given mask array
    convolution_1D(m, mask, n, length, maskLength);

    // get the details regarding the end time of this program and store it in the end struct
    clock_gettime(CLOCK_REALTIME, &end);

    // calculate exec time in microseconds
    time_t execTime = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;

    // output exec time
    printf("Execution time: %d microseconds.", execTime);

    // exit program
    return 0;
}


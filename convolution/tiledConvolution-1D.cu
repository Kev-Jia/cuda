// include necessary libs
#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

// define constant BLOCK_SIZE
#define BLOCK_SIZE 1024

// CUDA kernel function
__global__ void tiledConvolution_1D_Kernel(float* d_m, const float* __restrict__ d_mask, float* d_n, size_t length, size_t maskLength, int N_TILE_LENGTH)
{
    // define and initialize the variable where the resulting element of the convolution operation will be calculated and stored
    // this is to minimize writes to global memory
    // as automatic variables are stored in register memory
    float result = 0;

    // define and initialize indexing variables for input and result arrays - this is for brevity
    int n_index = blockIdx.x * N_TILE_LENGTH + threadIdx.x;
    int m_index = n_index - maskLength / 2;

    // define shared memory input array tile
    __shared__ float tile_m[BLOCK_SIZE];

    // if the input array index variable is within the bounds of the input array
    // then load the elements of d_m into their respective positions in the tile
    // otherwise just set the element of the tile to 0 (the element becomes a "ghost" element)
    if(m_index >= 0 && m_index < length)
    {
        tile_m[threadIdx.x] = d_m[m_index];
    }
    else
    {
        tile_m[threadIdx.x] = 0;
    }
    
    // sync all the threads in the block so faster threads don't work with uninitialized memory
    __syncthreads();
    
    // only allow a certain amount of threads per block to participate in calculating the result variable
    // because we only need to calculate N_TILE_LENGTH elements
    // < and not <= because of 0-based indexing
    if(threadIdx.x < N_TILE_LENGTH && n_index < length)
    {
        // calculate value of result element
        for(int i = 0; i < maskLength; ++i)
        {
            result += d_mask[i] * tile_m[threadIdx.x + i];
        }
        
        // write result variable to corresponding element of result array
        d_n[n_index] = result;
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

// host function that calls the CUDA kernel
void convolution_1D(float* m, float* mask, float* n, size_t length, size_t maskLength, int N_TILE_LENGTH)
{
    // define and initialize dimension variables containing data regarding the dimensions of the grid and the dimensions of each block
    dim3 numOfBlocks(ceil(length / (float) N_TILE_LENGTH), 1, 1);
    dim3 numOfThreads(BLOCK_SIZE, 1, 1);

    // define and initialize variables containing the number of bytes in each array
    size_t bytes_m = length * sizeof(float);
    size_t bytes_mask = maskLength * sizeof(float);

    // define the pointers that will point to the start of allocated device memory for each array
    float* d_m;
    float* d_mask;
    float* d_n;

    // allocate global memory for each array on the device and check for CUDA errors
    // input bytes variable is used for result array because both arrays have the same length
    cudaMalloc((void**) &d_m, bytes_m);
    errorCheck(__LINE__);
    cudaMalloc((void**) &d_mask, bytes_mask);
    errorCheck(__LINE__);
    cudaMalloc((void**) &d_n, bytes_m);
    errorCheck(__LINE__);

    // copy the data of each array to allocated global memory on the device and check for CUDA errors
    cudaMemcpy(d_m, m, bytes_m, cudaMemcpyHostToDevice);
    errorCheck(__LINE__);
    cudaMemcpy(d_mask, mask, bytes_mask, cudaMemcpyHostToDevice);
    errorCheck(__LINE__);

    // call the CUDA kernel and check for CUDA errors
    tiledConvolution_1D_Kernel<<<numOfBlocks, numOfThreads>>>(d_m, d_mask, d_n, length, maskLength, N_TILE_LENGTH);
    errorCheck(__LINE__);
    
    // copy the data of the result array from global memory to host DRAM and check for CUDA errors
    cudaMemcpy(n, d_n, bytes_m, cudaMemcpyDeviceToHost);
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
    
    // initialize pseudo-random number generator with seed of current seconds since 01/01/1970
    srand(time(NULL));
    
    // define and initialize size variables for each array
    // the input and result arrays have the same size and thus share a size variable
    // int instead of size_t for result tile length because otherwise typecasting to float will cause errors in the host function that calls the kernel
    size_t length = rand() % 1048577 + 15728640;
    size_t maskLength = 2 * (rand() % 64 + 192) + 1;
    int N_TILE_LENGTH = BLOCK_SIZE - (maskLength - 1);

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
        mask[j] = rand() % 1001 / 1000.0;
    }

    // perform 1D convolution operation on input array m using a given mask array
    convolution_1D(m, mask, n, length, maskLength, N_TILE_LENGTH);
    
    // get the details regarding the end time of this program and store it in the end struct
    clock_gettime(CLOCK_REALTIME, &end);

    // calculate exec time in microseconds
    time_t execTime = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;

    // output exec time
    printf("Execution time: %d microseconds.", execTime);

    // exit program
    return 0;
}


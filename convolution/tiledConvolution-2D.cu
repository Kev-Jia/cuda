// include necessary libs
#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

// define constant BLOCK_WIDTH
#define BLOCK_WIDTH 32

// CUDA kernel function
__global__ void tiledConvolution_2D_Kernel(float* d_m, const float* __restrict__ d_mask, float* d_n, size_t a, size_t b, size_t maskWidth, int N_TILE_WIDTH)
{
    // define and initialize the variable where the resulting element of the convolution operation will be calculated and stored
    // this is to minimize writes to global memory
    // as automatic variables are stored in register memory
    float result = 0;
    
    // define and initialize the variables that will be used for indexing - this is for brevity
    int n_row = blockIdx.y * N_TILE_WIDTH + threadIdx.y;
    int n_col = blockIdx.x * N_TILE_WIDTH + threadIdx.x;
    
    int m_row = n_row - maskWidth / 2;
    int m_col = n_col - maskWidth / 2;
    
    // define shared memory input array tile
    __shared__ float tile_m[BLOCK_WIDTH][BLOCK_WIDTH];
    
    // if the input array index variables are within the bounds of the input array
    // then load the elements of d_m into their respective positions in the tile
    // otherwise just set the element of the tile to 0 (the element becomes a "ghost" element)
    if(m_row >= 0 && m_row < a && m_col >= 0 && m_col < b)
    {
        tile_m[threadIdx.y][threadIdx.x] = d_m[m_row * b + m_col];
    }
    else
    {
        tile_m[threadIdx.y][threadIdx.x] = 0;
    }
    
    // sync all the threads in the block so faster threads don't work with uninitialized memory
    __syncthreads();
    
    // only allow a certain amount of threads per block to participate in calculating the result variable
    // because we only need to calculate N_TILE_LENGTH elements
    // < and not <= because of 0-based indexing
    if(threadIdx.y < N_TILE_WIDTH && threadIdx.x < N_TILE_WIDTH && n_row < a && n_col < b)
    {
        // calculate value of result element
        for(int i = 0; i < maskWidth; ++i)
        {
            for(int j = 0; j < maskWidth; ++j)
            {
                result += d_mask[i * maskWidth + j] * tile_m[threadIdx.y + i][threadIdx.x + j];
            }
        }
        
        // write result variable to corresponding element of result array
        d_n[n_row * b + n_col] = result;
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
void convolution_2D(float* m, float* mask, float* n, size_t a, size_t b, size_t maskWidth, int N_TILE_WIDTH)
{
    // define and initialize dimension variables containing data regarding the dimensions of the grid and the dimensions of each block
    dim3 numOfBlocks(ceil(b / (float) N_TILE_WIDTH), ceil(a / (float) N_TILE_WIDTH), 1);
    dim3 numOfThreads(BLOCK_WIDTH, BLOCK_WIDTH, 1);

    // define and initialize variables containing the number of bytes in each array
    size_t bytes_m = a * b * sizeof(float);
    size_t bytes_mask = maskWidth * maskWidth * sizeof(float);

    // define the pointers that will point to the start of allocated device memory for each array
    float* d_m;
    float* d_mask;
    float* d_n;

    // allocate global memory for each array on the device and check for CUDA errors
    // input bytes variable is used for result array because cuda-memcheck 0 errors but illegal memory accessboth arrays have the same length
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
    tiledConvolution_2D_Kernel<<<numOfBlocks, numOfThreads>>>(d_m, d_mask, d_n, a, b, maskWidth,  N_TILE_WIDTH);
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
    
    // define and initialize dimension variables for each array
    // the input and result arrays have the same dimensions and thus share dimension variables
    // int instead of size_t for result tile width because otherwise typecasting to float will cause errors in the host function that calls the kernel
    size_t a = rand() % 257 + 3840;
    size_t b = rand() % 257 + 3840;
    size_t maskWidth = 11;
    
    int N_TILE_WIDTH = BLOCK_WIDTH - (maskWidth - 1);

    // dynamically allocate DRAM memory for the arrays to account for them perhaps being too big to be statically allocated
    float* m = (float*) malloc(a * b * sizeof(float));
    float* mask = (float*) malloc(maskWidth * maskWidth * sizeof(float));
    float* n = (float*) malloc(a * b * sizeof(float));

    // assign a pseudo-random integer value from -64 to 64 for each element in input array m
    for(int i = 0; i < a * b; ++i)
    {
        m[i] = rand() % 129 - 64;
    }
  
    // assign a pseudo-random float value from 0 to 1 with a precision of 3 decimal places for each element in mask array
    for(int j = 0; j < maskWidth * maskWidth; ++j)
    {
        mask[j] = rand() % 1001 / 1000.0;
    }

    // perform 2D convolution operation on input array m using a given mask array
    convolution_2D(m, mask, n, a, b, maskWidth, N_TILE_WIDTH);
    
    // get the details regarding the end time of this program and store it in the end struct
    clock_gettime(CLOCK_REALTIME, &end);

    // calculate exec time in microseconds
    time_t execTime = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;

    // output exec time
    printf("Execution time: %d microseconds.", execTime);

    // exit program
    return 0;
}

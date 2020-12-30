// include necessary libs
#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

// CUDA kernel function
__global__ void convolution_2D_Kernel(float* d_m, float* d_mask, float* d_n, size_t a, size_t b, size_t maskWidth)
{
    // define and initialize the variable that will be used for indexing
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // define and initialize the variable that will allow the mask to align correctly with input array d_m
    // integer division is fine because it always truncates
    // we will only be using odd values for maskWidth anyway
    // the truncation of integer division will provide the perfect offset for aligning the centre element of the mask with the target d_m element
    int m_row = j - maskWidth / 2;
    int m_col = i - maskWidth / 2;

    // only have threads within the range of the length of the result array calculate an element of it
    // this is to prevent global memory segfaults
    if(i < b && j < a)
    {
        // iterate through all the elements in the mask
        // here's where the actual convolution operation will occur
        for(int k = 0; k < maskWidth; ++k)
        {
            for(int l = 0; l < maskWidth; ++l)
            {
                // this if statement is a boundary condition check
                // it checks whether the indexes needed for the convolution operation are within the bounds of the input array
                // if they are not
                // extra "ghost" elements past the beginning and end of the input array are used
                // these elements are set to 0 in our code
                // this ghost element stuff is done implicitly
                // as there is no need to do it explicitly, since 0 does not change the resulting element of the convolution operation on a specific element
                // we just leave the accumulating result of the convolution operation alone if the indexes are out of bounds
                if(m_row + l >= 0 && m_row + l < a && m_col + k >= 0 && m_col + k < b)
                {
                    // if the boundary check is satisfied
                    // then accumulate one part of the convolution operation to the result element of the result d_n array
                    // the convolution operation is done column by column to allow the global memory accesses to be coalesced
                    // this allows us to increase memory bandwidth
                    d_n[j * b + i] += d_m[(m_row + l) * b + m_col + k] * d_mask[l * maskWidth + k];
                }
            }
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

// host function that calls the CUDA kernel
void convolution_2D(float* m, float* mask, float* n, size_t a, size_t b, size_t maskWidth)
{
    // define and initialize dimension variables containing data regarding the dimensions of the grid and the dimensions of each block
    dim3 numOfBlocks(ceil(b / 32.0), ceil(a / 32.0), 1);
    dim3 numOfThreads(32, 32, 1);

    // define and initialize variables containing the number of bytes in each array
    size_t bytes_m = a * b * sizeof(float);
    size_t bytes_mask = maskWidth * maskWidth * sizeof(float);
    size_t bytes_n = a * b * sizeof(float);

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
    convolution_2D_Kernel<<<numOfBlocks, numOfThreads>>>(d_m, d_mask, d_n, a, b, maskWidth);
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

    // initialize pseudo-random number generator with seed of current seconds since 01/01/1970
    srand(time(NULL));
    
    // define and initialize dimension variables for each array
    // the input and result arrays have the same dimensions and thus share dimension variables
    size_t a = rand() % 513 + 7680;
    size_t b = rand() % 513 + 7680;
    size_t maskWidth = 2 * (rand() % 7 + 1) + 1;

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
    convolution_2D(m, mask, n, a, b, maskWidth);
    
    // get the details regarding the end time of this program and store it in the end struct
    clock_gettime(CLOCK_REALTIME, &end);

    // calculate exec time in microseconds
    time_t execTime = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;

    // output exec time
    printf("Execution time: %d microseconds.", execTime);

    // exit program
    return 0;
}

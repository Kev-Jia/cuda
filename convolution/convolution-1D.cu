// include necessary libs
#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

// CUDA kernel function
__global__ void convolution_1D_Kernel(float* d_m, float* d_mask, float* d_n, size_t length, size_t maskLength)
{
    // define and initialize the variable that will be used for indexing
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // define and initialize the variable that will allow the mask to align correctly with input array d_m
    // integer division is fine because it always truncates
    // we will only be using odd values for maskLength anyway
    // the truncation of integer division will provide the perfect offset for aligning the centre element of the mask with the target d_m element
    int m_index = i - (maskLength / 2);

    // initialize the to-be-generated result element of result array d_n to be 0 ready for accumulation in the for loop directly below
    d_n[i] = 0;

    // iterate through all the elements in the mask
    // here's where the actual convolution operation will occur
    for(int j = 0; j < maskLength; ++j)
    {
        // this if statement is a boundary condition check
        // it checks whether the indexes needed for the convolution operation are within the bounds of the input array
        // if they are not
        // extra "ghost" elements past the beginning and end of the input array are used
        // these elements are set to 0 in our code
        // this ghost element stuff is done implicitly
        // as there is no need to do it explicitly, since 0 does not change the resulting element of the convolution operation on a specific element
        // we just leave the accumulating result of the convolution operation alone if the indexes are out of bounds
        if(m_index + j >= 0 && m_index + j < length)
        {
            // if the boundary check is satisfied
            // then accumulate one part of the convolution operation to the result element of the result d_n array
            d_n[i] += d_m[m_index + j] * d_mask[j];
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
void convolution_1D(float* m, float* mask, float* n, size_t length, size_t maskLength)
{
    // define and initialize dimension variables containing data regarding the dimensions of the grid and the dimensions of each block
    dim3 numOfBlocks(ceil(length / 1024.0), 1, 1);
    dim3 numOfThreads(1024, 1, 1);

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
    convolution_1D_Kernel<<<numOfBlocks, numOfThreads>>>(d_m, d_mask, d_n, length, maskLength);
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
    
    // define and initialize size variables for each array
    // the input and result arrays have the same size and thus share a size variable
    size_t length = rand() % 1048577 + 15728640;
    size_t maskLength = 2 * (rand() % 64 + 192) + 1;

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

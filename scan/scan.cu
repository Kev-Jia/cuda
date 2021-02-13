#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>



// CUDA error checking
void errorCheck(unsigned int line)
{
    cudaError_t cudaError = cudaGetLastError();

    if(cudaError != cudaSuccess)
    {
        printf("CUDA error in line %u in file %s: %s\n", line - 1, __FILE__, cudaGetErrorString(cudaError));
        exit(EXIT_FAILURE);
    }
}


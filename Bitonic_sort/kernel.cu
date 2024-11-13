#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm>

#define THREADS_PER_BLOCK 512
#define SHARED_SIZE_LIMIT 1024

// =========================================================================

// Comparator: 0 denotes to be sorted in ASC order
__device__ inline void compare_exchange(int &v1, int &v2, int dir)
{
    int temp;
    // v1 occurs before v2 and v1 > v2 (1)
    // so v1, v2 are in DESC order, but we want them in INC (0)
    if ((v1 > v2) != dir)
    {
        temp = v1;
        v1 = v2;
        v2 = temp;
    }
}

// =========================================================================

// GPU Kernel Implementation of Bitonic Sort (Preprocessing)
// Divide the array into subarrays of sizes SHARED_SIZE and sort these bitonic arrays
// This way for the later kernel calls, we only care about the subarrays of sizes
// more than SHARED_SIZE. This preprocessing reduces the kernel calls by a lot
__global__ void bitonicSortPreprocess (int *gpuArray)
{   
    // Here we divide the gpuArray into arrays of size SHARED_SIZE
    // Since every instance fits in the size, it can be sorted
    // After the processing is complete, we only care about the instances that
    // DO NOT fit in the shared memory

    // start with declaring a shared memory
    __shared__ int shared_mem [SHARED_SIZE_LIMIT];

    // calculating the global index k
    int k = (SHARED_SIZE_LIMIT * blockIdx.x) + threadIdx.x;

    // now we load the contents into the shared memory
    // since half of the threads are launched inside a shared memory block,
    // each threads loads twice. Since threadIdx.x < SHARED_SIZE/2 we can also load threadIdx.x + SHARED_SIZE/2.
    // Also here the blockDim is SHARED_SIZE last blockIdx : n/SHARED_SIZE -1 
    // so k is atmost SHARED_SIZE (n/SHARED_SIZE - 1) + SHARED_SIZE/2 -1 which is k <= n - SHARED_SIZE/2 - 1

    shared_mem[threadIdx.x] = gpuArray[k];
    shared_mem[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = gpuArray[k + (SHARED_SIZE_LIMIT / 2)];

    // 0 indicates ascending, 1 indicates descending
    int dir;

    for(int i = 2 ; i <= SHARED_SIZE_LIMIT ; i = i << 1)
    {   
        dir = (threadIdx.x & (i / 2)) != 0;
        // rank
        for (int j = i >> 1; j >= 1; j = j >> 1)
        {
            // before we do swap, at each loop the rank changes and so do the shared indices
            // we need to sync to prevent W-W conflicts
            __syncthreads();
            // any thread that goes to idx, gets the value from index : 2*idx - (idx & (rank - 1))
            int sh_ind1 = 2 * threadIdx.x - (threadIdx.x & (j - 1));
            int sh_ind2 = sh_ind1 + j;
            // the values at these indices are swapped as per the direction
            compare_exchange(shared_mem[sh_ind1], shared_mem[sh_ind2], dir);
        }
    }

    dir = (blockIdx.x & 1) != 0;

    for (int j = SHARED_SIZE_LIMIT >> 1; j >= 1; j = j >> 1)
    {
       __syncthreads();
        // any thread that goes to idx, gets the value from index : 2*idx - (idx & (rank - 1))
        int sh_ind1 = 2 * threadIdx.x - (threadIdx.x & (j - 1));
        int sh_ind2 = sh_ind1 + j;
        // the values at these indices are swapped as per the direction
        compare_exchange(shared_mem[sh_ind1], shared_mem[sh_ind2], dir);
    }

    // before writing values, make sure everything is synced
    __syncthreads();

    // write to global shared_mem
    gpuArray[k] = shared_mem[threadIdx.x];
    gpuArray[k + (SHARED_SIZE_LIMIT / 2)] = shared_mem[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];

}

// =========================================================================

// GPU Implementation of Bitonic Sort using shared memory for sorting
// gpu arrays for a given j that fits in the shared memory
// This reduces global memory access and kernel
__global__ void bitonicSortShared (int *gpuArray, int i, int n)
{
    // global index is k (k < n/2)
    // If a subarray of size j fits in the shared memory, all subarrays with j >> x for some x,
    // will also fit in the shared memory. Hence all j >> x are taken care of
    // inside this kernel. We do not need to rely on host
    // this also decreases the kernel launches.

    // start with declaring a shared memory
    __shared__ int shared_mem [SHARED_SIZE_LIMIT];

    // calculating the global index k
    int k = (SHARED_SIZE_LIMIT * blockIdx.x) + threadIdx.x;

    // direction is given by the current size value (subarray) and global index
    int dir = (((blockIdx.x * blockDim.x + threadIdx.x) & (i / 2)) != 0);

    // now we load the contents into the shared memory
    // since half of the threads are launched inside a shared memory block,
    // each threads loads twice. Since threadIdx.x < SHARED_SIZE/2
    // we can also load threadIdx.x + SHARED_SIZE/2.
    // Also here the blockDim is SHARED_SIZE
    // last blockIdx : n/SHARED_SIZE -1 
    // so k is atmost SHARED_SIZE (n/SHARED_SIZE - 1) + SHARED_SIZE/2 -1
    // which is k <= n - SHARED_SIZE/2 - 1

    shared_mem[threadIdx.x] = gpuArray[k];
    shared_mem[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = gpuArray[k + (SHARED_SIZE_LIMIT / 2)];

    // Now, the block size or "i" is the SHARED_SIZE
    // So all new_j are calculated accordingly
    for(int rank = SHARED_SIZE_LIMIT / 2; rank >= 1; rank = rank >> 1){
        // before we do swap, at each loop the rank changes and so do the shared indices
        // we need to sync to prevent W-W conflicts
        __syncthreads();
        // any thread that goes to idx, gets the value from index : 2*idx - (idx & (rank - 1))
        int sh_ind1 = 2 * threadIdx.x - (threadIdx.x & (rank - 1));
        int sh_ind2 = sh_ind1 + rank;
        // the values at these indices are swapped as per the direction
        compare_exchange(shared_mem[sh_ind1], shared_mem[sh_ind2], dir);
    }

    // before writing values, make sure everything is synced
    __syncthreads();

    // write to global memory
    gpuArray[k] = shared_mem[threadIdx.x];
    gpuArray[k + (SHARED_SIZE_LIMIT / 2)] = shared_mem[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];

}

// =========================================================================

//GPU Kernel Implementation of Bitonic Sort using Global Memory
__global__ void bitonicSortGlobal (int *gpuArray, int i, int j, int n)
{
    // global index is k (k < n/2)
    // pos = 2k - (k & (j-1)) denotes the element in gpuArray it points to
    // pos + j is the index of next element

    int k = threadIdx.x + (blockDim.x * blockIdx.x);
    int ind1 = (2 * k) - (k & (j-1));
    int ind2 = ind1 + j;

    int dir = (k & (i / 2)) != 0;
    compare_exchange(gpuArray[ind1], gpuArray[ind2], dir);

}

// =========================================================================


int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <array_size>\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);

    srand(time(NULL));

    // ======================================================================
    // arCpu contains the input random array
    // arrSortedGpu should contain the sorted array copied from GPU to CPU
    // ======================================================================
    int* arrCpu = (int*)malloc(size * sizeof(int));
    int* arrSortedGpu = (int*)malloc(size * sizeof(int));

    for (int i = 0; i < size; i++) {
        arrCpu[i] = rand() % 1000;
    }

    float gpuTime, h2dTime, d2hTime, cpuTime = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // ======================================================================
    // Transfer data (arr_cpu) to device
    // ======================================================================

    // your code goes here .......

    int newSize = 1 << (int)(log2(size) + 1);
    // get the difference between current size and next power of 2 
    int sizeDiff = newSize - size;

    // pad with these many zeroes
    int *gpuArray;
    cudaMalloc(&gpuArray, newSize * sizeof(int));
    // Pinning arrCpu for lower H2D time
    cudaHostRegister((void *)arrCpu, size * sizeof(int), cudaHostRegisterDefault);
    // setting the first sizeDiff elements as -1
    cudaMemset(gpuArray, -1, sizeDiff * sizeof(int));
    // copying from cpu to gpu memory
    cudaMemcpy(gpuArray + sizeDiff, arrCpu, size * sizeof(int), cudaMemcpyHostToDevice); 

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&h2dTime, start, stop);

    cudaEventRecord(start);
    
    // ======================================================================
    // Perform bitonic sort on GPU
    // ======================================================================

    // your code goes here .......

    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = newSize / THREADS_PER_BLOCK;

    // for all subarray of sizes <= SHARED_SIZE, preprocess all subarrays that fit in shared memory
    // this is done to reduce kernel launches later
    bitonicSortPreprocess <<< newSize / SHARED_SIZE_LIMIT, SHARED_SIZE_LIMIT / 2 >>> (gpuArray);

    // Since ize of subarray is a power of 2, after processing SHARED_SIZE sized subarrays,
    // the next available length is double of SHARED_SIZE_LIMIT
    // Now the for loop starts with SHARED_SIZE_LIMIT * 2 which reduces the number of kernels
    for (int i = SHARED_SIZE_LIMIT << 1 ; i <= newSize; i = i << 1)
    {   
        for(int j = i >> 1; j >= 1; j = j >> 1)
        {   
            if (j < SHARED_SIZE_LIMIT)
            {
                bitonicSortShared <<< newSize / SHARED_SIZE_LIMIT, SHARED_SIZE_LIMIT / 2 >>> (gpuArray, i, newSize);
                break;
            }
            else
            {
                bitonicSortGlobal <<< blocksPerGrid, threadsPerBlock / 2 >>> (gpuArray, i, j, newSize);
            }
        }
    }
    // after the kernel is launched, CPU waits for kernel to finish
    // Kernel launches are non blocking and control is given back to CPU
    // hence, memory can be pinned in parallel
    cudaHostRegister((void *)(arrSortedGpu), size * sizeof(int), cudaHostRegisterDefault);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    cudaEventRecord(start);

    // ======================================================================
    // Transfer sorted data back to host (copied to arr_sorted_gpu)
    // ======================================================================

    // your code goes here .......

    cudaMemcpy(arrSortedGpu, gpuArray + sizeDiff, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2hTime, start, stop);

    auto startTime = std::chrono::high_resolution_clock::now();
    
    // CPU sort for performance comparison
    std::sort(arrCpu, arrCpu + size);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    cpuTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    cpuTime = cpuTime / 1000;

    int match = 1;
    for (int i = 0; i < size; i++) {
        if (arrSortedGpu[i] != arrCpu[i]) {
            match = 0;
            break;
        }
    }

    free(arrCpu);
    free(arrSortedGpu);

    if (match)
        printf("\033[1;32mFUNCTIONAL SUCCESS\n\033[0m");
    else {
        printf("\033[1;31mFUNCTIONCAL FAIL\n\033[0m");
        return 0;
    }
    
    printf("\033[1;34mArray size         :\033[0m %d\n", size);
    printf("\033[1;34mCPU Sort Time (ms) :\033[0m %f\n", cpuTime);
    float gpuTotalTime = h2dTime + gpuTime + d2hTime;
    int speedup = (gpuTotalTime > cpuTime) ? (gpuTotalTime/cpuTime) : (cpuTime/gpuTotalTime);
    float meps = size / (gpuTotalTime * 0.001) / 1e6;
    printf("\033[1;34mGPU Sort Time (ms) :\033[0m %f\n", gpuTotalTime);
    printf("\033[1;34mGPU Sort Speed     :\033[0m %f million elements per second\n", meps);
    if (gpuTotalTime < cpuTime) {
        printf("\033[1;32mPERF PASSING\n\033[0m");
        printf("\033[1;34mGPU Sort is \033[1;32m %dx \033[1;34mfaster than CPU !!!\033[0m\n", speedup);
        printf("\033[1;34mH2D Transfer Time (ms):\033[0m %f\n", h2dTime);
        printf("\033[1;34mKernel Time (ms)      :\033[0m %f\n", gpuTime);
        printf("\033[1;34mD2H Transfer Time (ms):\033[0m %f\n", d2hTime);
    } else {
        printf("\033[1;31mPERF FAILING\n\033[0m");
        printf("\033[1;34mGPU Sort is \033[1;31m%dx \033[1;34mslower than CPU, optimize further!\n", speedup);
        printf("\033[1;34mH2D Transfer Time (ms):\033[0m %f\n", h2dTime);
        printf("\033[1;34mKernel Time (ms)      :\033[0m %f\n", gpuTime);
        printf("\033[1;34mD2H Transfer Time (ms):\033[0m %f\n", d2hTime);
        return 0;
    }

    return 0;
}


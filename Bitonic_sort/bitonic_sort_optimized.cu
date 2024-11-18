#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cmath>
#include <chrono>

#define THREADS_PER_BLOCK 512 // Optimal threads per block
#define SHARED_SIZE_LIMIT 1024 // Shared memory limit

// Efficient helper to get the next power of two
__host__ int getNextPowerOfTwo(int n) {
    if ((n & (n - 1)) == 0) return n; // Already a power of two
    while (n & (n - 1)) {
        n = n & (n - 1);
    }
    return n << 1;
}

// Device comparator for bitonic sort
__device__ inline void compare_exchange(int &v1, int &v2, int direction) {
    if ((v1 > v2) == direction) {
        int temp = v1;
        v1 = v2;
        v2 = temp;
    }
}

// CUDA kernel for shared memory-based bitonic sort
__global__ void bitonicSortKernelShared(int *arr, int n) {
    extern __shared__ int sharedArr[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load elements into shared memory
    if (idx < n) {
        sharedArr[tid] = arr[idx];
    } else {
        sharedArr[tid] = INT_MAX; // Padding
    }

    __syncthreads();

    // Perform the bitonic sort within shared memory
    for (int size = 2; size <= blockDim.x; size <<= 1) {
        int dir = ((tid & size) == 0);
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            int index = tid ^ stride;
            if (index > tid) {
                compare_exchange(sharedArr[tid], sharedArr[index], dir);
            }
            __syncthreads();
        }
    }

    // Write sorted data back to global memory
    if (idx < n) {
        arr[idx] = sharedArr[tid];
    }
}

// CUDA kernel for global memory-based bitonic sort
__global__ void bitonicSortKernelGlobal(int *arr, int n, int j, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        int ixj = idx ^ j;
        if (ixj > idx) {
            if ((idx & k) == 0 && arr[idx] > arr[ixj]) {
                int temp = arr[idx];
                arr[idx] = arr[ixj];
                arr[ixj] = temp;
            } else if ((idx & k) != 0 && arr[idx] < arr[ixj]) {
                int temp = arr[idx];
                arr[idx] = arr[ixj];
                arr[ixj] = temp;
            }
        }
    }
}

// CPU implementation of bitonic sort for comparison
void bitonicSortCPU(int *arr, int n) {
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = 0; i < n; i++) {
                int ixj = i ^ j;
                if (ixj > i) {
                    if ((i & k) == 0 && arr[i] > arr[ixj]) {
                        int temp = arr[i];
                        arr[i] = arr[ixj];
                        arr[ixj] = temp;
                    } else if ((i & k) != 0 && arr[i] < arr[ixj]) {
                        int temp = arr[i];
                        arr[i] = arr[ixj];
                        arr[ixj] = temp;
                    }
                }
            }
        }
    }
}

void fillArray(int *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 100; // Random values between 0 and 99
    }
}

int main(int argc, char *argv[]) {
    int n = 1024;
    if (argc >= 2) n = atoi(argv[1]);

    // Adjust array size to the next power of two
    int paddedSize = getNextPowerOfTwo(n);
    printf("Original Size: %d, Padded Size: %d\n", n, paddedSize);

    // Allocate and initialize host memory
    int *arrCPU = (int *)malloc(paddedSize * sizeof(int));
    int *arrGPU;
    fillArray(arrCPU, n);

    // CPU Timing
    auto cpuStart = std::chrono::high_resolution_clock::now();
    bitonicSortCPU(arrCPU, paddedSize);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpuDuration = cpuEnd - cpuStart;
    double cpuTime = cpuDuration.count();
    printf("CPU Execution Time: %.6f seconds\n", cpuTime);

    // Allocate device memory and copy data
    cudaMalloc((void **)&arrGPU, paddedSize * sizeof(int));
    cudaMemcpy(arrGPU, arrCPU, paddedSize * sizeof(int), cudaMemcpyHostToDevice);

    // GPU Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim((paddedSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    printf("Threads per Block: %d, Blocks per Grid: %d\n", THREADS_PER_BLOCK, gridDim.x);

    // Shared memory phase
    bitonicSortKernelShared<<<gridDim, blockDim, THREADS_PER_BLOCK * sizeof(int)>>>(arrGPU, paddedSize);
    cudaDeviceSynchronize();

    // Global memory phase
    for (int k = 2; k <= paddedSize; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonicSortKernelGlobal<<<gridDim, blockDim>>>(arrGPU, paddedSize, j, k);
            cudaDeviceSynchronize();
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Measure GPU execution time
    float gpuTime;
    cudaEventElapsedTime(&gpuTime, start, stop);
    gpuTime /= 1000; // Convert to seconds
    printf("GPU Execution Time: %.6f seconds\n", gpuTime);

    // Calculate and display speedup
    double speedup = cpuTime / gpuTime;
    printf("Speedup (CPU/GPU): %.2fx\n", speedup);

    // Validate result
    cudaMemcpy(arrCPU, arrGPU, paddedSize * sizeof(int), cudaMemcpyDeviceToHost);
    bool sorted = true;
    for (int i = 1; i < n; i++) {
        if (arrCPU[i - 1] > arrCPU[i]) {
            sorted = false;
            break;
        }
    }

    printf("Array Sorted: %s\n", sorted ? "True" : "False");

    // Cleanup
    free(arrCPU);
    cudaFree(arrGPU);

    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <string.h>

#define N 1024       // Number of elements in the array
#define BLOCK_SIZE 256 // CUDA block size



// CUDA kernel for Bitonic Sort
__global__ void bitonicSortKernel(int *arr, int n, int j, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx;

    if (i < n) {
        int ixj = i ^ j;

        if (ixj > i) {
            if ((i & k) == 0) {
                if (arr[i] > arr[ixj]) {
                    int temp = arr[i];
                    arr[i] = arr[ixj];
                    arr[ixj] = temp;
                }
            }
            if ((i & k) != 0) {
                if (arr[i] < arr[ixj]) {
                    int temp = arr[i];
                    arr[i] = arr[ixj];
                    arr[ixj] = temp;
                }
            }
        }
    }
}

// CPU implementation of Bitonic Sort
void bitonicSortCPU(int *arr, int n) {
    int i, j, k;
    for (k = 2; k <= n; k <<= 1) {
        for (j = k >> 1; j > 0; j >>= 1) {
            for (i = 0; i < n; i++) {
                int ixj = i ^ j;
                if (ixj > i) {
                    if ((i & k) == 0 && arr[i] > arr[ixj]) {
                        int temp = arr[i];
                        arr[i] = arr[ixj];
                        arr[ixj] = temp;
                    }
                    if ((i & k) != 0 && arr[i] < arr[ixj]) {
                        int temp = arr[i];
                        arr[i] = arr[ixj];
                        arr[ixj] = temp;
                    }
                }
            }
        }
    }
}

// Helper function to fill the array with reverse sorted values
void fillArrayReverseSorted(int *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = n - i - 1; // Fill the array in descending order
    }
}

// Helper function to fill the array with uniformly distributed values
void fillArrayUniform(int *arr, int n) {
    int range = 10000;
    for (int i = 0; i < n; i++) {
        arr[i] = (i % range) + (rand() % (range / 10));
    }
    // Shuffle the array (Fisher-Yates)
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}

// Function to generate a random Gaussian number
double generateGaussian(double mean, double stddev) {
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return z0 * stddev + mean;
}

// Function to generate an array of Gaussian numbers
void generateGaussianArray(int *array, int size, double mean, double stddev) {
    for (int i = 0; i < size; i++) {
        array[i] = (int)generateGaussian(mean, stddev);
    }
}

// Helper function to check if the array is sorted
int isSorted(int *arr1, int *arr2, int n) {
    for (int i = 0; i < n; i++) {
        if (arr1[i] != arr2[i]) {
            return 0; // Not identically sorted
        }
    }
    return 1; // Identically sorted
}

// Function to test Bitonic Sort on CPU and GPU
void testArray(int *originalArray, int n, int threadsPerBlock, int numBlocks) {
    int *arrCPU = (int *)malloc(n * sizeof(int));
    int *arrGPU, *d_arr;
    

    // Allocate GPU memory
    cudaMalloc((void **)&d_arr, n * sizeof(int));
    

    clock_t start, end;

    // Copy result back to host
    arrGPU = (int *)malloc(n * sizeof(int));

    // CPU Bitonic Sort
    start = clock();

    memcpy(arrCPU, originalArray, n * sizeof(int)); // Keep the original array unchanged
    bitonicSortCPU(arrCPU, n);

    end = clock();
    double cpuTime = (double)(end - start) / CLOCKS_PER_SEC;

    // GPU Bitonic Sort
    start = clock();

    cudaMemcpy(d_arr, originalArray, n * sizeof(int), cudaMemcpyHostToDevice);

    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonicSortKernel<<<numBlocks, threadsPerBlock>>>(d_arr, n, j, k);
            cudaDeviceSynchronize();
        }
    }
    cudaMemcpy(arrGPU, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    end = clock();

    double gpuTime = (double)(end - start) / CLOCKS_PER_SEC;
       

    // Verify results and calculate speedup
    if (isSorted(arrGPU, arrCPU, n)) {
        printf("CPU Time: %lf seconds\n", cpuTime);
        printf("GPU Time: %lf seconds\n", gpuTime);
        printf("Speedup (CPU/GPU): %lfx\n", cpuTime / gpuTime);
    } else {
        printf("Sorting failed on either CPU or GPU!\n");
    }

    // Free memory
    cudaFree(d_arr);
    free(arrCPU);
    free(arrGPU);
}

int main(int argc, char *argv[]) {
    // Default values
    int n = N;
    int threadsPerBlock = BLOCK_SIZE;
    int numBlocks = 4;

    // Parse command-line arguments
    if (argc >= 2) n = atoi(argv[1]);
    if (argc >= 3) threadsPerBlock = atoi(argv[2]);
    if (argc >= 4) numBlocks = atoi(argv[3]);

    printf("Array Size: %d, Threads per Block: %d, Number of Blocks: %d\n\n", n, threadsPerBlock, numBlocks);

    int *arr = (int *)malloc(n * sizeof(int));

    // Test reverse sorted array
    printf("Testing reverse sorted array:\n");
    fillArrayReverseSorted(arr, n);
    testArray(arr, n, threadsPerBlock, numBlocks);

    // Test uniform distribution array
    printf("\nTesting uniform distribution array:\n");
    fillArrayUniform(arr, n);
    testArray(arr, n, threadsPerBlock, numBlocks);

    // Test Gaussian distribution array
    printf("\nTesting Gaussian distribution array:\n");
    generateGaussianArray(arr, n, n/2, n/6); // Mean=50, StdDev=10
    testArray(arr, n, threadsPerBlock, numBlocks);

    // Free memory
    free(arr);
    return 0;
}

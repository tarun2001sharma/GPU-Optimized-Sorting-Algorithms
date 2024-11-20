#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>

#define MAX_STACK_SIZE 1024

// CUDA QuickSort Kernel
__global__ void quicksortKernel(int *data, int left, int right) {
    __shared__ int stack[MAX_STACK_SIZE];
    int top = -1;

    if (threadIdx.x == 0) {
        // Push initial bounds onto the stack
        top++;
        stack[top] = left;
        top++;
        stack[top] = right;
    }

    while (top >= 0) {
        int r = stack[top--];
        int l = stack[top--];

        int i = l;
        int j = r;
        int pivot = data[(l + r) / 2];
        int temp;

        while (i <= j) {
            while (data[i] < pivot) i++;
            while (data[j] > pivot) j--;
            if (i <= j) {
                temp = data[i];
                data[i] = data[j];
                data[j] = temp;
                i++;
                j--;
            }
        }

        if (l < j) {
            top++;
            stack[top] = l;
            top++;
            stack[top] = j;
        }
        if (i < r) {
            top++;
            stack[top] = i;
            top++;
            stack[top] = r;
        }
    }
}

// CPU QuickSort
void quicksortCPU(int *arr, int left, int right) {
    if (left >= right) return;

    int pivot = arr[right];
    int l = left;
    int r = right - 1;
    int temp;

    while (l <= r) {
        while (l <= r && arr[l] <= pivot) l++;
        while (l <= r && arr[r] >= pivot) r--;
        if (l < r) {
            temp = arr[l];
            arr[l] = arr[r];
            arr[r] = temp;
        }
    }

    temp = arr[l];
    arr[l] = arr[right];
    arr[right] = temp;

    quicksortCPU(arr, left, l - 1);
    quicksortCPU(arr, l + 1, right);
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

// Helper function to fill array with reverse sorted values
void fillArrayReverseSorted(int *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = n - i - 1;
    }
}

// Helper function to generate a Gaussian array
double generateGaussian(double mean, double stddev) {
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return z0 * stddev + mean;
}

void generateGaussianArray(int *arr, int n, double mean, double stddev) {
    for (int i = 0; i < n; i++) {
        arr[i] = (int)generateGaussian(mean, stddev);
    }
}

// Check if array is sorted
int isSorted(int *arr1, int *arr2, int n) {
    for (int i = 0; i < n; i++) {
        if (arr1[i] != arr2[i]) {
            return 0; // Not identically sorted
        }
    }
    return 1; // Identically sorted
}

// Test QuickSort on CPU and GPU
void testArray(int *originalArray, int n) {
    int *arrCPU = (int *)malloc(n * sizeof(int));
    int *arrGPU, *d_arr;
    

    cudaMalloc((void **)&d_arr, n * sizeof(int));
    arrGPU = (int *)malloc(n * sizeof(int));
    

    // CPU QuickSort
    clock_t startCPU = clock();
    memcpy(arrCPU, originalArray, n * sizeof(int));
    quicksortCPU(arrCPU, 0, n - 1);
    clock_t endCPU = clock();
    double cpuTime = (double)(endCPU - startCPU) / CLOCKS_PER_SEC;

    // GPU QuickSort
    clock_t startGPU = clock();
    cudaMemcpy(d_arr, originalArray, n * sizeof(int), cudaMemcpyHostToDevice);
    quicksortKernel<<<1, 1>>>(d_arr, 0, n - 1);
    cudaDeviceSynchronize(); // Ensure kernel execution is complete
    cudaMemcpy(arrGPU, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    clock_t endGPU = clock();
    double gpuTime = (double)(endGPU - startGPU) / CLOCKS_PER_SEC;

    
    

    if (isSorted(arrCPU, arrGPU, n)) {
        printf("CPU Time: %f seconds\n", cpuTime);
        printf("GPU Time: %f seconds\n", gpuTime);
        printf("Speedup: %fx\n", cpuTime / gpuTime);
    } else {
        printf("Sorting failed on either CPU or GPU!\n");
    }

    cudaFree(d_arr);
    free(arrCPU);
    free(arrGPU);
}


int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <array_size>\n", argv[0]);
        return -1;
    }

    int n = atoi(argv[1]); // Parse array size from command-line arguments
    if (n <= 0) {
        printf("Array size must be a positive integer.\n");
        return -1;
    }

    int *array = (int *)malloc(n * sizeof(int));

    // Choose data type for testing
    fillArrayUniform(array, n);
    printf("Testing QuickSort with Uniform Data:\n");
    testArray(array, n);

    fillArrayReverseSorted(array, n);
    printf("Testing QuickSort with Reverse Sorted Data:\n");
    testArray(array, n);

    generateGaussianArray(array, n, n/2, n/6);
    printf("Testing QuickSort with Gaussian Data:\n");
    testArray(array, n);

    free(array);
    return 0;
}

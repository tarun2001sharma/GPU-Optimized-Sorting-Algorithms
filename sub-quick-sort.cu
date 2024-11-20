
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

// Global variables to track comparisons and swaps for the CPU quicksort
int numberOfComparisons = 0;
int numberOfSwaps = 0;

// Helper function to fill the array with random values
void fillArray(int *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 100; // Random values between 0 and 99
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

// CPU QuickSort Implementation (provided version)
void quick_sort(int array[], int left, int right) {
    int i, j, pivot, y;
    i = left;
    j = right;

    pivot = array[(i + j) / 2];

    while (i <= j) {
        while (array[i] < pivot && i < right) {
            i++;
            numberOfComparisons++;
        }
        while (array[j] > pivot && j > left) {
            j--;
            numberOfComparisons++;
        }
        if (i <= j) {
            y = array[i];
            array[i] = array[j];
            array[j] = y;
            i++;
            j--;
            numberOfSwaps++;
        }
    }
    if (j > left) {
        quick_sort(array, left, j);
    }
    if (i < right) {
        quick_sort(array, i, right);
    }
}

// Selection sort used in GPU quicksort when depth is too large or size is small
__device__ void selection_sort(int *data, int left, int right) {
    for (int i = left; i <= right; ++i) {
        int min_val = data[i];
        int min_idx = i;
        for (int j = i + 1; j <= right; ++j) {
            if (data[j] < min_val) {
                min_val = data[j];
                min_idx = j;
            }
        }
        if (i != min_idx) {
            int temp = data[i];
            data[i] = data[min_idx];
            data[min_idx] = temp;
        }
    }
}

// GPU QuickSort Kernel
__global__ void cdp_simple_quicksort(int *data, int left, int right, int depth) {
    if (depth >= 16 || right - left <= 32) {
        selection_sort(data, left, right);
        return;
    }

    int *lptr = data + left;
    int *rptr = data + right;
    int pivot = data[(left + right) / 2];

    while (lptr <= rptr) {
        while (*lptr < pivot) lptr++;
        while (*rptr > pivot) rptr--;
        if (lptr <= rptr) {
            int temp = *lptr;
            *lptr = *rptr;
            *rptr = temp;
            lptr++;
            rptr--;
        }
    }

    int nright = rptr - data;
    int nleft = lptr - data;

    if (left < (rptr - data)) {
        cdp_simple_quicksort<<<1, 1>>>(data, left, nright, depth + 1);
    }
    if ((lptr - data) < right) {
        cdp_simple_quicksort<<<1, 1>>>(data, nleft, right, depth + 1);
    }
}

// GPU QuickSort Wrapper
void gpu_qsort(int *data, int n) {
    int *gpuData;
    cudaMalloc((void **)&gpuData, n * sizeof(int));
    cudaMemcpy(gpuData, data, n * sizeof(int), cudaMemcpyHostToDevice);

    cdp_simple_quicksort<<<1, 1>>>(gpuData, 0, n - 1, 0);
    cudaDeviceSynchronize();

    cudaMemcpy(data, gpuData, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(gpuData);
}

int main(int argc, char *argv[]) {
    int n = 1024;
    if (argc >= 2) n = atoi(argv[1]);

    printf("Array Size: %d\n", n);

    int *arrCPU, *arrGPU;
    clock_t start, end;

    arrCPU = (int *)malloc(n * sizeof(int));
    arrGPU = (int *)malloc(n * sizeof(int));

    fillArray(arrCPU, n);
    
    start = clock();
    memcpy(arrGPU, arrCPU, n * sizeof(int));
    quick_sort(arrCPU, 0, n - 1);
    end = clock();

    double cpuTime = (double)(end - start) / CLOCKS_PER_SEC;
    printf("CPU Time: %lf seconds\n", cpuTime);
    printf("Number of Comparisons (CPU): %d\n", numberOfComparisons);
    printf("Number of Swaps (CPU): %d\n", numberOfSwaps);


    start = clock();
    gpu_qsort(arrGPU, n);
    end = clock();
    double gpuTime = (double)(end - start) / CLOCKS_PER_SEC;
    printf("GPU Time: %lf seconds\n", gpuTime);

    if (!isSorted(arrGPU, arrCPU, n)) {
        printf("GPU sorting failed!\n");
        return 1;
    }

    double speedup = cpuTime / gpuTime;
    printf("Speedup (CPU/GPU): %lfx\n", speedup);

    free(arrCPU);
    free(arrGPU);

    return 0;
}

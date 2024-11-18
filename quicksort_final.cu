#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

int N;  // Size of the array, now a variable

// Helper function to fill the array with random values
void fillArray(int *arr, int n) {
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 100; // Random values between 0 and 99
    }
}

// Helper function to check if the array is sorted
int isSorted(int *arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        if (arr[i] > arr[i + 1]) return 0;
    }
    return 1;
}

// CPU QuickSort implementation
void quicksort_cpu(int *arr, int left, int right) {
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

    quicksort_cpu(arr, left, l - 1);
    quicksort_cpu(arr, l + 1, right);
}

// GPU QuickSort kernel
__global__ void quicksort_gpu(int *data, int left, int right) {
    // Stack for storing left and right indices
    __shared__ int stack[1024];
    int top = -1;

    // Only one thread is used in this kernel for simplicity
    if (threadIdx.x == 0) {
        // Push initial values
        top++;
        stack[top] = left;
        top++;
        stack[top] = right;

        while (top >= 0) {
            // Pop right and left
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

            // Push left side to stack
            if (l < j) {
                top++;
                stack[top] = l;
                top++;
                stack[top] = j;
            }

            // Push right side to stack
            if (i < r) {
                top++;
                stack[top] = i;
                top++;
                stack[top] = r;
            }
        }
    }
}

int main(int argc, char **argv) {
    // Check if array size is provided
    if (argc != 2) {
        printf("Usage: %s <array_size>\n", argv[0]);
        return -1;
    }

    // Parse array size
    N = atoi(argv[1]);
    if (N <= 0) {
        printf("Array size must be a positive integer.\n");
        return -1;
    }

    int *h_arr, *h_arr_cpu, *h_arr_gpu;
    int *d_arr;
    size_t size = N * sizeof(int);

    // Allocate memory
    h_arr = (int*)malloc(size);
    h_arr_cpu = (int*)malloc(size);
    h_arr_gpu = (int*)malloc(size);
    cudaMalloc((void**)&d_arr, size);

    // Fill array with random data
    fillArray(h_arr, N);
    memcpy(h_arr_cpu, h_arr, size);
    memcpy(h_arr_gpu, h_arr, size);

    // CPU QuickSort
    clock_t start_cpu = clock();
    quicksort_cpu(h_arr_cpu, 0, N - 1);
    clock_t end_cpu = clock();
    double cpu_time = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC * 1000.0;

    // Check if CPU array is sorted
    if (isSorted(h_arr_cpu, N))
        printf("CPU sorting successful.\n");
    else
        printf("CPU sorting failed.\n");

    // Copy data to device
    cudaMemcpy(d_arr, h_arr_gpu, size, cudaMemcpyHostToDevice);

    // GPU QuickSort
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);
    quicksort_gpu<<<1, 1>>>(d_arr, 0, N - 1);
    cudaEventRecord(stop_gpu);

    cudaEventSynchronize(stop_gpu);
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);

    // Copy data back to host
    cudaMemcpy(h_arr_gpu, d_arr, size, cudaMemcpyDeviceToHost);

    // Check if GPU array is sorted
    if (isSorted(h_arr_gpu, N))
        printf("GPU sorting successful.\n");
    else
        printf("GPU sorting failed.\n");

    // Print execution times
    printf("CPU time: %f ms\n", cpu_time);
    printf("GPU time: %f ms\n", gpu_time);

    printf("Speedup: %fx\n", cpu_time / gpu_time);

    // Free memory
    free(h_arr);
    free(h_arr_cpu);
    free(h_arr_gpu);
    cudaFree(d_arr);

    return 0;
}

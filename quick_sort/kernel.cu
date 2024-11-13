#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define THREADS_PER_BLOCK 256
#define MAX_DEPTH 16  // Maximum depth to allow recursive calls in GPU

// Host utility function for pivot selection (Median of three method)
__device__ int medianOfThree(int* data, int left, int right) {
    int mid = left + (right - left) / 2;
    int a = data[left], b = data[mid], c = data[right];
    if ((a < b) != (a < c)) return left;
    if ((b < a) != (b < c)) return mid;
    return right;
}

// Partition kernel
__global__ void partitionKernel(int* data, int left, int right, int* leftCount, int* rightCount, int pivot) {
    int index = left + threadIdx.x + blockIdx.x * blockDim.x;
    if (index <= right) {
        if (data[index] < pivot) atomicAdd(leftCount, 1);
        else atomicAdd(rightCount, 1);
    }
}

// Parallel quicksort function using a stack-based approach
__global__ void quicksortKernel(int* data, int* stackLeft, int* stackRight, int* stackTop) {
    __shared__ int leftCount;
    __shared__ int rightCount;

    while (true) {
        int top = atomicSub(stackTop, 1);
        if (top < 0) break;

        int left = stackLeft[top];
        int right = stackRight[top];
        if (right - left <= 1) continue;

        // Select pivot using the median of three
        int pivotIndex = medianOfThree(data, left, right);
        int pivot = data[pivotIndex];

        // Initialize shared memory counters
        if (threadIdx.x == 0) {
            leftCount = 0;
            rightCount = 0;
        }
        __syncthreads();

        // Partition the array in parallel
        partitionKernel<<<(right - left + THREADS_PER_BLOCK) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            data, left, right, &leftCount, &rightCount, pivot);

        cudaDeviceSynchronize();

        // Calculate the split position based on partition counts
        int pivotPos = left + leftCount;

        // Move elements to the left and right of pivot
        __syncthreads();
        for (int i = left + threadIdx.x; i < right; i += blockDim.x) {
            if (data[i] < pivot && i >= pivotPos) atomicExch(&data[i], data[pivotPos++]);
            else if (data[i] >= pivot && i < pivotPos) atomicExch(&data[i], data[pivotPos--]);
        }

        // Push left and right subarrays onto the stack for further sorting
        if (pivotPos - left > 1) {
            int pos = atomicAdd(stackTop, 1);
            stackLeft[pos] = left;
            stackRight[pos] = pivotPos - 1;
        }
        if (right - pivotPos > 1) {
            int pos = atomicAdd(stackTop, 1);
            stackLeft[pos] = pivotPos + 1;
            stackRight[pos] = right;
        }
    }
}

// Host function to initialize quicksort
void quicksort(int* data, int n) {
    int *d_data, *stackLeft, *stackRight, *stackTop;

    cudaMalloc(&d_data, n * sizeof(int));
    cudaMalloc(&stackLeft, MAX_DEPTH * sizeof(int));
    cudaMalloc(&stackRight, MAX_DEPTH * sizeof(int));
    cudaMalloc(&stackTop, sizeof(int));

    cudaMemcpy(d_data, data, n * sizeof(int), cudaMemcpyHostToDevice);

    int hostTop = 0;
    cudaMemcpy(stackTop, &hostTop, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(stackLeft, &hostTop, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(stackRight, &hostTop, sizeof(int), cudaMemcpyHostToDevice);

    // Initial push of entire array onto stack
    cudaMemcpy(stackLeft, &hostTop, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(stackRight, &n, sizeof(int), cudaMemcpyHostToDevice);

    // Launch quicksort kernel
    quicksortKernel<<<1, THREADS_PER_BLOCK>>>(d_data, stackLeft, stackRight, stackTop);
    cudaDeviceSynchronize();

    // Copy the sorted array back to the host
    cudaMemcpy(data, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_data);
    cudaFree(stackLeft);
    cudaFree(stackRight);
    cudaFree(stackTop);
}

// Utility function to print the array
void printArray(int* arr, int size) {
    for (int i = 0; i < size; i++) printf("%d ", arr[i]);
    printf("\n");
}

// Test the implementation
int main() {
    int n = 16;  // Size of the array
    int data[n];

    // Random initialization of array
    srand(time(0));
    for (int i = 0; i < n; i++) data[i] = rand() % 100;

    printf("Original array:\n");
    printArray(data, n);

    // Run quicksort on the GPU
    quicksort(data, n);

    printf("Sorted array:\n");
    printArray(data, n);

    return 0;
}

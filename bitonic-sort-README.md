
# Bitonic Sort (CUDA Implementation)

CUDA implementation of QuickSort with Selection Sort fallback for GPU-accelerated sorting. Includes CPU QuickSort for performance comparison.

## Compile and Run

### Compile
```bash
nvcc -o bitonic-sort bitonic-sort.cu -lm -rdc=true
```

### Run
```bash
./bitonic-sort <size_of_input> <threads/optional> <blocks/optional>
```
- `<size_of_input>`: Size of the input array.
- `<threads>`: Number of threads per block (optional).
- `<blocks>`: Number of blocks (optional).

**Note:** Ensure `threads × blocks >= size_of_input`. If not provided, defaults are calculated automatically.

### Example
```bash
./bitonic-sort 2048 128 16
```

## Outputs
- Array size.
- CPU and GPU execution times.
- Sorting accuracy check.
- CPU-to-GPU speedup.

## Requirements
- CUDA-capable GPU with NVIDIA CUDA Toolkit.
- GCC or compatible C compiler.

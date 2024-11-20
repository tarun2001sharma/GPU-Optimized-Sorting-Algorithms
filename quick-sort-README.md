
# GPU QuickSort with Selection Sort Fallback

CUDA-based QuickSort implementation with a Selection Sort fallback for small subarrays. Includes a CPU QuickSort for comparison.

## Compile and Run

### Compile
```bash
nvcc -o quick-sort quick-sort.cu -lm -rdc=true
```

### Run
```bash
./quick-sort <size_of_input>
```
- `<size_of_input>`: Size of the array to sort. Defaults to `1024`.

### Example
```bash
./quick-sort 2048
```

## Outputs
- Array size.
- CPU and GPU execution times.
- Sorting accuracy validation.
- Speedup (CPU vs. GPU).

## Requirements
- CUDA-capable GPU with NVIDIA CUDA Toolkit.
- GCC or compatible C compiler.

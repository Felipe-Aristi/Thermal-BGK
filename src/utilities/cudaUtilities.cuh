
#ifndef CUDA_UTILITIES_CUH
#define CUDA_UTILITIES_CUH

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

// ============================================================
// CUDA error checking
// ============================================================

inline void cudaCheck(const cudaError_t error, const char *file, const int line) noexcept
{
    if (error != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA error: %s\nFile: %s\nLine: %d\n", cudaGetErrorString(error), file, line);
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) cudaCheck((call), __FILE__, __LINE__)

// ============================================================
// Check the most recent CUDA kernel launch
// ============================================================

inline void cudaCheckLastError(const char *file, const int line) noexcept
{
    cudaCheck(cudaGetLastError(), file, line);
}

#define CUDA_KERNEL_CHECK() cudaCheckLastError(__FILE__, __LINE__)

#endif // CUDA_UTILITIES_CUH

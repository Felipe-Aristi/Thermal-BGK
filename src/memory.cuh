
#ifndef MEMORY_CUH
#define MEMORY_CUH

#include <cuda_runtime.h>

#include <stdexcept>
#include <utility>

#include "constants.cuh"
#include "utilities/cudaUtilities.cuh"

// ============================================================
// Population buffers
//
// A: Current populations
// B: Next populations
// ============================================================

struct Memory
{
    real_t *A = nullptr;
    real_t *B = nullptr;
};

// ============================================================
// Allocate GPU memory
// ============================================================

inline void allocateMemory(Memory &memory)
{
    CUDA_CHECK(cudaMalloc(&memory.A, bytesPopulations));
    CUDA_CHECK(cudaMalloc(&memory.B, bytesPopulations));
}

// ============================================================
// Swap population buffers
// ============================================================

inline void swapMemory(Memory &memory)
{
    std::swap(memory.A, memory.B);
}

// ============================================================
// Free GPU memory
// ============================================================

inline void freeMemory(Memory &memory)
{
    cudaFree(memory.A);
    cudaFree(memory.B);

    memory.A = nullptr;
    memory.B = nullptr;
}

#endif // MEMORY_CUH


#ifndef LAUNCH_CUH
#define LAUNCH_CUH

#include <cuda_runtime.h>

#include "constants.cuh"
#include "memory.cuh"
#include "kernels.cuh"

#include "utilities/cudaUtilities.cuh"

#include "../cases/Rb/initialization.cuh"
#include "../cases/Rb/bc.cuh"

// ============================================================
// Initialize Rayleigh-Benard populations

// ============================================================

inline void launchRBInitialization(Memory &memory,
                                   const real_t rho0,
                                   const real_t T0,
                                   const real_t deltaT,
                                   const real_t m,
                                   const real_t epsilon,
                                   const real_t ay)
{
    CUDA_CHECK(cudaMemset(memory.A, 0, bytesPopulations));
    CUDA_CHECK(cudaMemset(memory.B, 0, bytesPopulations));

    const dim3 block(blockX, blockY, 1);

    const dim3 grid(
        (NXPhysical + blockX - 1) / blockX,
        (NY + blockY - 1) / blockY,
        1
    );

    initializeRBKernel<D2V37><<<grid, block>>>(
        memory.A,
        rho0,
        T0,
        deltaT,
        m,
        epsilon,
        ay
    );

    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================
// Thermal diffuse-reflection boundaries
//
// One thread per horizontal wall column.
//
// grid.y = 0 -> lower wall
// grid.y = 1 -> upper wall
//
// Ghost populations are written to buffer A.
// ============================================================

inline void launchRBDiffuseBoundary(Memory &memory,
                                    const real_t TLower,
                                    const real_t TUpper)
{
    const dim3 block(blockX, 1, 1);

    const dim3 grid(
        (NXPhysical + blockX - 1) / blockX,
        2,
        1
    );

    diffuseWallKernel<D2V37><<<grid, block>>>(
        memory.A,
        TLower,
        TUpper
    );

    CUDA_KERNEL_CHECK();
}

// ============================================================
// Thermal BGK evolution
//
// Pull streaming + collision.
//
// Read:  A
// Write: B
//
// Constant dynamic viscosity:
//
// tau = 0.5 + mu0 / (rho * TH)
// ============================================================

inline void launchThermalBGK(Memory &memory,
                             const real_t mu0,
                             const real_t ax,
                             const real_t ay)
{
    const dim3 block(blockX, blockY, 1);

    const dim3 grid(
        (NXPhysical + blockX - 1) / blockX,
        (NY + blockY - 1) / blockY,
        1
    );

    thermalBGKKernel<D2V37><<<grid, block>>>(
        memory.A,
        memory.B,
        mu0,
        ax,
        ay
    );

    CUDA_KERNEL_CHECK();
}

// ============================================================
// Complete Rayleigh-Benard timestep
// ============================================================

inline void launchRBStep(Memory &memory,
                         const real_t mu0,
                         const real_t ax,
                         const real_t ay,
                         const real_t TLower,
                         const real_t TUpper)
{
    launchRBDiffuseBoundary(memory, TLower, TUpper);

    launchThermalBGK(memory, mu0, ax, ay);

    swapMemory(memory);
}

#endif // LAUNCH_CUH

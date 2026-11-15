
#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>

#include "constants.cuh"
#include "lbm.cuh"
#include "utilities/indexing.cuh"
#include "utilities/constexprFor.cuh"

// ============================================================
// Thermal BGK kernel
// ============================================================

template <typename Stencil>
__global__ void thermalBGKKernel(const real_t *__restrict__ A,
                                 real_t *__restrict__ B,
                                 const real_t mu0,
                                 const real_t ax,
                                 const real_t ay)
{
    const label_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const label_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= NXPhysical || !Domain::insideFluidY(static_cast<int>(y))) return;

    // ========================================================
    // Pull streaming
    // ========================================================

    real_t f[Stencil::Q];

    constexprFor<0, Stencil::Q>([&](auto Index)
    {
        constexpr label_t I = decltype(Index)::value;

        f[I] = A[pullID<Stencil, I>(x, y)];
    });

    // ========================================================
    // Macroscopic reconstruction
    // ========================================================

    const Macroscopic macro = recoverMacroscopic<Stencil>(f, ax, ay);

    // ========================================================
    // Local relaxation time
    // ========================================================

    const real_t tau = 0.5 + mu0 / (macro.rho * macro.TH);

    // ========================================================
    // Thermal BGK collision
    // ========================================================

    collideBGK<Stencil>(f, macro, tau, ax, ay);

    // ========================================================
    // Store updated populations
    // ========================================================

    constexprFor<0, Stencil::Q>([&](auto Index)
    {
        constexpr label_t I = decltype(Index)::value;

        B[popID<I>(x, y)] = f[I];
    });
}

#endif // KERNELS_CUH

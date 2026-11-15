
#ifndef RB_INITIALIZATION_CUH
#define RB_INITIALIZATION_CUH

#include <cuda_runtime.h>
#include <cmath>

#include "../../src/constants.cuh"
#include "../../src/lbm.cuh"
#include "../../src/utilities/indexing.cuh"
#include "../../src/utilities/constexprFor.cuh"

// ============================================================
// Rayleigh-Benard initialization
//
// Conductive temperature profile:
//
// T(y) = T0 + deltaT * (0.5 - eta)
//
// Hydrostatic density:
//
// rho(y) = rho0 * (T(y) / T0)^m
//
// eta = (y - lowerWallY) / H
//
// Initial hydrodynamic velocity:
//
// uxH = 0
// uyH = 0
// ============================================================

template <typename Stencil>
__global__ void initializeRBKernel(real_t *__restrict__ A,
                                   const real_t rho0,
                                   const real_t T0,
                                   const real_t deltaT,
                                   const real_t m,
                                   const real_t epsilon,
                                   const real_t ay)
{
    // ========================================================
    // Spatial coordinates
    // ========================================================

    const label_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const label_t y = blockIdx.y * blockDim.y + threadIdx.y;

    // ========================================================
    // Physical fluid region
    // ========================================================

    if (x >= NXPhysical || !Domain::insideFluidY(y)) return;

    // ========================================================
    // Dimensionless vertical coordinate
    //
    // eta = 0 at the lower wall
    // eta = 1 at the upper wall
    // ========================================================

    const real_t eta = (static_cast<real_t>(y) - Domain::lowerWallY) / Domain::height;

    // ========================================================
    // Conductive temperature profile
    // ========================================================

    const real_t TBase = T0 + deltaT * (0.5 - eta);

    // ========================================================
    // Hydrostatic density profile
    // ========================================================

    const real_t rhoBase = rho0 * pow(TBase / T0, m);

    // ========================================================
    // Initial thermal perturbation
    // ========================================================

    const real_t kx = 2.0 * pi / static_cast<real_t>(NXPhysical);

    const real_t perturbation = epsilon * T0 *
                                cos(kx * static_cast<real_t>(x)) *
                                sin(pi * eta);

    const real_t TH = TBase + perturbation;

    // ========================================================
    // Initial density
    //
    // Preserve the hydrostatic pressure profile:
    //
    // rho * TH = rhoBase * TBase
    // ========================================================

    const real_t rho = rhoBase * TBase / TH;

    // ========================================================
    // Convert hydrodynamic fields to lattice fields
    //
    // uH = uL + a / 2
    // TH = TL + a^2 / (4 * D)
    //
    // Initially, uH = 0.
    // ========================================================

    const real_t uxL = 0.0;
    const real_t uyL = -0.5 * ay;

    const real_t TL = TH - 0.25 * ay * ay /
                      static_cast<real_t>(Stencil::D);

    const real_t usqL = uxL * uxL + uyL * uyL;

    // ========================================================
    // Initialize complete equilibrium populations
    //
    // No population shift fi - wi.
    // ========================================================

    real_t f[Stencil::Q];

    constexprFor<0, Stencil::Q>([&](auto index)
    {
        constexpr label_t I = decltype(index)::value;

        constexpr real_t cx = static_cast<real_t>(Stencil::template cx<I>());
        constexpr real_t cy = static_cast<real_t>(Stencil::template cy<I>());

        const real_t cu = cx * uxL + cy * uyL;

        f[I] = feq<Stencil, I>(rho, TL, cu, usqL);
    });

    // A is streamed before the first collision, so it must contain
    // post-collision populations, just as it does after each buffer swap.
    const Macroscopic macro = recoverMacroscopic<Stencil>(f, 0.0, ay);
    const real_t tau = 0.5 + dynamicViscosityReference / (macro.rho * macro.TH);
    collideBGK<Stencil>(f, macro, tau, 0.0, ay);

    constexprFor<0, Stencil::Q>([&](auto index)
    {
        constexpr label_t I = decltype(index)::value;
        A[popID<I>(x, y)] = f[I];
    });
}

#endif // RB_INITIALIZATION_CUH

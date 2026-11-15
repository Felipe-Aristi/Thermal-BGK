
#ifndef RB_BC_CUH
#define RB_BC_CUH

#include <cuda_runtime.h>

#include "../../src/constants.cuh"
#include "../../src/pdf.cuh"
#include "../../src/utilities/indexing.cuh"
#include "../../src/utilities/constexprFor.cuh"

// ============================================================
// Thermal diffuse-reflection boundary condition
//
// Lower wall: normal points in +y
// Upper wall: normal points in -y
//
// Full populations fi are used.
//
// One horizontal column shares one wall normalization.
//
// All wall-crossing links are included.
// ============================================================

template <typename Stencil, bool LowerWall>
__device__ __forceinline__
void diffuseWallColumn(real_t *A,
                       const label_t x,
                       const real_t Tw) noexcept
{
    real_t outgoing = 0.0;
    real_t referenceFlux = 0.0;

    // ========================================================
    // Calculate outgoing mass and reference incoming flux
    // ========================================================

    constexprFor<0, Domain::ghostLayers>([&](auto Ghost)
    {
        constexpr int k = static_cast<int>(decltype(Ghost)::value);

        constexpr label_t ghostY = LowerWall
            ? Domain::fluidYBegin - 1 - k
            : Domain::fluidYEnd + 1 + k;

        constexprFor<0, Stencil::Q>([&](auto Index)
        {
            constexpr label_t I = decltype(Index)::value;

            constexpr int cx = Stencil::template cx<I>();
            constexpr int cy = Stencil::template cy<I>();

            // =================================================
            // Populations leaving the physical fluid
            //
            // Lower wall: cy < -k
            // Upper wall: cy >  k
            // =================================================

            if constexpr (LowerWall ? (cy < -k) : (cy > k))
            {
                constexpr label_t sourceY =
                    static_cast<label_t>(static_cast<int>(ghostY) - cy);

                const label_t sourceX = pullX<cx>(x);

                outgoing += A[popID<I>(sourceX, sourceY)];
            }

            // =================================================
            // Populations emitted toward the physical fluid
            //
            // Lower wall: cy >  k
            // Upper wall: cy < -k
            // =================================================

            if constexpr (LowerWall ? (cy > k) : (cy < -k))
            {
                referenceFlux += feq<Stencil, I>(1.0, Tw, 0.0, 0.0);
            }
        });
    });

    // ========================================================
    // Diffuse-reflection normalization
    //
    // Mass emitted = mass leaving the fluid
    // ========================================================

    const real_t Aw = outgoing / referenceFlux;

    // ========================================================
    // Fill all three ghost layers
    //
    // Only directions that return to the fluid are nonzero.
    // ========================================================

    constexprFor<0, Domain::ghostLayers>([&](auto Ghost)
    {
        constexpr int k = static_cast<int>(decltype(Ghost)::value);

        constexpr label_t ghostY = LowerWall
            ? Domain::fluidYBegin - 1 - k
            : Domain::fluidYEnd + 1 + k;

        constexprFor<0, Stencil::Q>([&](auto Index)
        {
            constexpr label_t I = decltype(Index)::value;
            constexpr int cy = Stencil::template cy<I>();

            if constexpr (LowerWall ? (cy > k) : (cy < -k))
            {
                A[popID<I>(x, ghostY)] =
                    Aw * feq<Stencil, I>(1.0, Tw, 0.0, 0.0);
            }
            else
            {
                A[popID<I>(x, ghostY)] = 0.0;
            }
        });
    });
}

// ============================================================
// CUDA kernel
//
// blockIdx.y = 0: lower wall
// blockIdx.y = 1: upper wall
//
// One thread per wall column.
// ============================================================

template <typename Stencil>
__global__ void diffuseWallKernel(real_t *A,
                                  const real_t TLower,
                                  const real_t TUpper)
{
    const label_t x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= NXPhysical) return;

    if (blockIdx.y == 0)
    {
        diffuseWallColumn<Stencil, true>(A, x, TLower);
    }
    else
    {
        diffuseWallColumn<Stencil, false>(A, x, TUpper);
    }
}

#endif // RB_BC_CUH

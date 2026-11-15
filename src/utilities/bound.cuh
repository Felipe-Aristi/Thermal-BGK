
#ifndef BOUND_CUH
#define BOUND_CUH

#include <cuda_runtime.h>

#include "types.cuh"

// ============================================================
// Domain geometry and boundaries
//
// X: periodic
// Y: solid walls with ghost layers
//
// All geometrical quantities are compile-time constants.
// ============================================================

template <label_t Width, label_t Height, typename Stencil>
struct Bounds
{
    // ========================================================
    // Physical domain
    // ========================================================

    inline static constexpr label_t NXPhysical = Width;
    inline static constexpr label_t NYPhysical = Height;

    // ========================================================
    // Ghost layers
    // ========================================================

    inline static constexpr label_t ghostLayers = Stencil::maxAbsC;

    // ========================================================
    // Allocated domain
    // ========================================================

    inline static constexpr label_t NX = NXPhysical;
    inline static constexpr label_t NY = NYPhysical + 2 * ghostLayers;

    // ========================================================
    // Fluid limits
    // ========================================================

    inline static constexpr label_t fluidYBegin = ghostLayers;
    inline static constexpr label_t fluidYEnd = fluidYBegin + NYPhysical - 1;

    // ========================================================
    // Ghost-layer limits
    // ========================================================

    inline static constexpr label_t lowerGhostYBegin = 0;
    inline static constexpr label_t lowerGhostYEnd = fluidYBegin - 1;

    inline static constexpr label_t upperGhostYBegin = fluidYEnd + 1;
    inline static constexpr label_t upperGhostYEnd = NY - 1;

    // ========================================================
    // Interior region for bulk streaming
    // ========================================================

    inline static constexpr label_t bulkYBegin = fluidYBegin + ghostLayers;
    inline static constexpr label_t bulkYEnd = fluidYEnd - ghostLayers;

    // ========================================================
    // Wall positions
    // ========================================================

    inline static constexpr real_t lowerWallY = static_cast<real_t>(fluidYBegin) - 0.5;
    inline static constexpr real_t upperWallY = static_cast<real_t>(fluidYEnd) + 0.5;

    inline static constexpr real_t height = upperWallY - lowerWallY;

    // ========================================================
    // Region checks
    // ========================================================

    __host__ __device__ static constexpr bool insideFluidY(const int y) noexcept
    {
        return y >= static_cast<int>(fluidYBegin) && y <= static_cast<int>(fluidYEnd);
    }

    __host__ __device__ static constexpr bool insideBulkY(const int y) noexcept
    {
        return y >= static_cast<int>(bulkYBegin) && y <= static_cast<int>(bulkYEnd);
    }

    __host__ __device__ static constexpr bool insideLowerGhostY(const int y) noexcept
    {
        return y >= static_cast<int>(lowerGhostYBegin) && y <= static_cast<int>(lowerGhostYEnd);
    }

    __host__ __device__ static constexpr bool insideUpperGhostY(const int y) noexcept
    {
        return y >= static_cast<int>(upperGhostYBegin) && y <= static_cast<int>(upperGhostYEnd);
    }

    __host__ __device__ static constexpr bool insideAllocatedY(const int y) noexcept
    {
        return y >= 0 && y < static_cast<int>(NY);
    }

    __host__ __device__ static constexpr bool insideBoundaryY(const int y) noexcept
    {
        return insideFluidY(y) && !insideBulkY(y);
    }
};

#endif // BOUND_CUH

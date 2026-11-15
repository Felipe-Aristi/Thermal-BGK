
#ifndef INDEXING_CUH
#define INDEXING_CUH

#include "../constants.cuh"

// ============================================================
// Spatial indexing
//
// id = x + y * NX
// ============================================================

[[nodiscard]] __host__ __device__ constexpr __forceinline__
label_t idx(const label_t x, const label_t y) noexcept
{
    return x + y * NX;
}

// ============================================================
// Population indexing (SoA)
//
// idf = I * Ncells + id
// ============================================================

template <label_t I>
[[nodiscard]] __host__ __device__ constexpr __forceinline__
std::size_t popID(const label_t id) noexcept
{
    return static_cast<std::size_t>(I) * Ncells + id;
}

template <label_t I>
[[nodiscard]] __host__ __device__ constexpr __forceinline__
std::size_t popID(const label_t x, const label_t y) noexcept
{
    return popID<I>(idx(x, y));
}

// ============================================================
// Periodic x indexing
//
// xs = x - cx
// ============================================================

template <int C>
[[nodiscard]] __host__ __device__ constexpr __forceinline__
label_t pullX(const label_t x) noexcept
{
    int source = static_cast<int>(x) - C;

    if (source < 0) source += static_cast<int>(NX);
    else if (source >= static_cast<int>(NX)) source -= static_cast<int>(NX);

    return static_cast<label_t>(source);
}

// ============================================================
// Nonperiodic y indexing
//
// ys = y - cy
//
// A signed integer prevents unsigned underflow.
// ============================================================

template <int C>
[[nodiscard]] __host__ __device__ constexpr __forceinline__
int pullY(const label_t y) noexcept
{
    return static_cast<int>(y) - C;
}

// ============================================================
// Check whether the pull source is a physical fluid node
// ============================================================

template <typename Stencil, label_t I>
[[nodiscard]] __host__ __device__ constexpr __forceinline__
bool pullSourceInsideFluid(const label_t y) noexcept
{
    constexpr int cy = Stencil::template cy<I>();

    return Domain::insideFluidY(pullY<cy>(y));
}

// ============================================================
// Check whether the pull source is in a lower ghost layer
// ============================================================

template <typename Stencil, label_t I>
[[nodiscard]] __host__ __device__ constexpr __forceinline__
bool pullSourceInLowerGhost(const label_t y) noexcept
{
    constexpr int cy = Stencil::template cy<I>();

    return Domain::insideLowerGhostY(pullY<cy>(y));
}

// ============================================================
// Check whether the pull source is in an upper ghost layer
// ============================================================

template <typename Stencil, label_t I>
[[nodiscard]] __host__ __device__ constexpr __forceinline__
bool pullSourceInUpperGhost(const label_t y) noexcept
{
    constexpr int cy = Stencil::template cy<I>();

    return Domain::insideUpperGhostY(pullY<cy>(y));
}

// ============================================================
// Check whether the pull source is allocated
// ============================================================

template <typename Stencil, label_t I>
[[nodiscard]] __host__ __device__ constexpr __forceinline__
bool pullSourceAllocated(const label_t y) noexcept
{
    constexpr int cy = Stencil::template cy<I>();

    return Domain::insideAllocatedY(pullY<cy>(y));
}

// ============================================================
// Pull-streaming population index
//
// Reads:
//
// A[I * Ncells + xs + ys * NX]
//
// xs = periodic(x - cx_i)
// ys = y - cy_i
// ============================================================

template <typename Stencil, label_t I>
[[nodiscard]] __host__ __device__ constexpr __forceinline__
std::size_t pullID(const label_t x, const label_t y) noexcept
{
    constexpr int cx = Stencil::template cx<I>();
    constexpr int cy = Stencil::template cy<I>();

    const label_t xs = pullX<cx>(x);
    const int ys = pullY<cy>(y);

    return popID<I>(xs, static_cast<label_t>(ys));
}

#endif // INDEXING_CUH


#ifndef PDF_CUH
#define PDF_CUH

#include "utilities/types.cuh"
#include "../stencil/d2v37.cuh"

// ============================================================
// Fourth-order thermal equilibrium
//
// fi^eq = wi * rho * (1 + Phi1 + Phi2 + Phi3 + Phi4)
//
// Arguments:
// rho : physical density
// T   : lattice or force-shifted temperature
// cu  : ci . u
// usq : u . u
//
// Gravity is incorporated by passing the force-shifted
// velocity and temperature to this function.
// ============================================================

template <typename Stencil, label_t I>
[[nodiscard]] __host__ __device__ __forceinline__
real_t feq(const real_t rho,
           const real_t T,
           const real_t cu,
           const real_t usq) noexcept
{
    constexpr real_t wi = Stencil::template w<I>();
    constexpr real_t c2 = static_cast<real_t>(Stencil::template c2<I>());
    constexpr real_t c4 = c2 * c2;

    constexpr real_t D = static_cast<real_t>(Stencil::D);
    constexpr real_t cs2 = Stencil::cs2;
    constexpr real_t cs4 = Stencil::cs4;

    const real_t theta = T - cs2;
    const real_t theta2 = theta * theta;

    const real_t cu2 = cu * cu;
    const real_t cu3 = cu2 * cu;
    const real_t cu4 = cu2 * cu2;
    const real_t usq2 = usq * usq;

    // ========================================================
    // First-order Hermite contribution
    // ========================================================

    const real_t Phi1 = Stencil::inv_cs2 * cu;

    // ========================================================
    // Second-order Hermite contribution
    // ========================================================

    const real_t Phi2 = static_cast<real_t>(0.5) * Stencil::inv_cs4 *
                        (cu2 - cs2 * usq + theta * (c2 - D * cs2));

    // ========================================================
    // Third-order Hermite contribution
    // ========================================================

    const real_t Phi3 = static_cast<real_t>(1.0 / 6.0) * Stencil::inv_cs6 *
                        (cu3 - static_cast<real_t>(3) * cs2 * usq * cu
                        + static_cast<real_t>(3) * theta * cu *
                        (c2 - (D + static_cast<real_t>(2)) * cs2));

    // ========================================================
    // Fourth-order Hermite contribution
    // ========================================================

    const real_t H4u = cu4 - static_cast<real_t>(6) * cs2 * usq * cu2
                       + static_cast<real_t>(3) * cs4 * usq2;

    const real_t H4uT = cu2 * (c2 - (D + static_cast<real_t>(4)) * cs2)
                        - cs2 * usq * (c2 - (D + static_cast<real_t>(2)) * cs2);

    const real_t H4T = c4 - static_cast<real_t>(2) * (D + static_cast<real_t>(2)) * cs2 * c2
                       + D * (D + static_cast<real_t>(2)) * cs4;

    const real_t Phi4 = static_cast<real_t>(1.0 / 24.0) * Stencil::inv_cs8 *
                        (H4u + static_cast<real_t>(6) * theta * H4uT
                        + static_cast<real_t>(3) * theta2 * H4T);

    return wi * rho * (static_cast<real_t>(1) + Phi1 + Phi2 + Phi3 + Phi4);
}

#endif // PDF_CUH

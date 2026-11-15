
#ifndef LBM_CUH
#define LBM_CUH

#include "pdf.cuh"
#include "utilities/types.cuh"
#include "utilities/constexprFor.cuh"

// ============================================================
// Macroscopic variables
//
// L: Lattice quantities
// H: Hydrodynamic quantities
// ============================================================

struct Macroscopic
{
    real_t rho = 0.0;

    real_t uxL = 0.0;
    real_t uyL = 0.0;
    real_t TL = 0.0;

    real_t uxH = 0.0;
    real_t uyH = 0.0;
    real_t TH = 0.0;
};

// ============================================================
// Recover macroscopic variables from full populations
//
// rho = sum_i fi
// rho * uL = sum_i ci * fi
// rho * (D * TL + uL^2) = sum_i ci^2 * fi
//
// Hydrodynamic corrections:
//
// uH = uL + a / 2
// TH = TL + a^2 / (4 * D)
// ============================================================

template <typename Stencil>
__host__ __device__ __forceinline__
Macroscopic recoverMacroscopic(const real_t (&f)[Stencil::Q],
                               const real_t ax,
                               const real_t ay) noexcept
{
    real_t rho = 0.0;
    real_t jx = 0.0;
    real_t jy = 0.0;
    real_t m2 = 0.0;

    constexprFor<0, Stencil::Q>([&](auto index)
    {
        constexpr label_t I = decltype(index)::value;

        constexpr real_t cx = static_cast<real_t>(Stencil::template cx<I>());
        constexpr real_t cy = static_cast<real_t>(Stencil::template cy<I>());
        constexpr real_t c2 = static_cast<real_t>(Stencil::template c2<I>());

        const real_t fi = f[I];

        rho += fi;
        jx += cx * fi;
        jy += cy * fi;
        m2 += c2 * fi;
    });

    const real_t invRho = 1.0 / rho;
    const real_t invD = 1.0 / static_cast<real_t>(Stencil::D);

    const real_t uxL = jx * invRho;
    const real_t uyL = jy * invRho;
    const real_t usqL = uxL * uxL + uyL * uyL;

    const real_t TL = (m2 * invRho - usqL) * invD;

    const real_t a2 = ax * ax + ay * ay;

    Macroscopic macro;

    macro.rho = rho;

    macro.uxL = uxL;
    macro.uyL = uyL;
    macro.TL = TL;

    macro.uxH = uxL + 0.5 * ax;
    macro.uyH = uyL + 0.5 * ay;
    macro.TH = TL + 0.25 * a2 * invD;

    return macro;
}

// ============================================================
// Thermal BGK collision with gravitational acceleration
//
// fi* = fi - omega * (fi - fieq)
//
// omega = 1 / tau
//
// Force-shifted equilibrium arguments:
//
// uBar = uL + tau * a
// TBar = TL + tau * (1 - tau) * a^2 / D
//
// Full populations are used.
// ============================================================

template <typename Stencil>
__host__ __device__ __forceinline__
void collideBGK(real_t (&f)[Stencil::Q],
                const Macroscopic &macro,
                const real_t tau,
                const real_t ax,
                const real_t ay) noexcept
{
    const real_t omega = 1.0 / tau;
    const real_t a2 = ax * ax + ay * ay;

    const real_t uxBar = macro.uxL + tau * ax;
    const real_t uyBar = macro.uyL + tau * ay;
    const real_t TBar = macro.TL + tau * (1.0 - tau) * a2 / static_cast<real_t>(Stencil::D);

    const real_t usqBar = uxBar * uxBar + uyBar * uyBar;

    constexprFor<0, Stencil::Q>([&](auto index)
    {
        constexpr label_t I = decltype(index)::value;

        constexpr real_t cx = static_cast<real_t>(Stencil::template cx<I>());
        constexpr real_t cy = static_cast<real_t>(Stencil::template cy<I>());

        const real_t cuBar = cx * uxBar + cy * uyBar;
        const real_t fieq = feq<Stencil, I>(macro.rho, TBar, cuBar, usqBar);

        f[I] -= omega * (f[I] - fieq);
    });
}

#endif // LBM_CUH

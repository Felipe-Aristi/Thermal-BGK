
#ifndef D2V37_CUH
#define D2V37_CUH

#include "../src/utilities/types.cuh"

#if defined(__CUDACC__)
#define D2V37_HD __host__ __device__
#else
#define D2V37_HD
#endif

// ============================================================
// D2V37 thermal velocity stencil
//
// All lattice parameters are evaluated at compile time.
//
// Velocity shells:
// (0,0)       ->  1 velocity
// (1,0)       ->  4 velocities
// (1,1)       ->  4 velocities
// (2,0)       ->  4 velocities
// (2,1)       ->  8 velocities
// (2,2)       ->  4 velocities
// (3,0)       ->  4 velocities
// (3,1)       ->  8 velocities
//
// Total: 37 velocities
// ============================================================

struct D2V37
{
    // ========================================================
    // Lattice dimensions
    // ========================================================

    inline static constexpr label_t Q = 37;
    inline static constexpr label_t D = 2;
    inline static constexpr int maxAbsC = 3;

    // ========================================================
    // Reference sound speed and temperature
    // ========================================================

    inline static constexpr real_t a =
        static_cast<real_t>(1.19697977039307435897238846385);

    inline static constexpr real_t cs =
        static_cast<real_t>(1) / a;

    inline static constexpr real_t cs2 = cs * cs;
    inline static constexpr real_t cs4 = cs2 * cs2;
    inline static constexpr real_t cs6 = cs4 * cs2;
    inline static constexpr real_t cs8 = cs4 * cs4;

    inline static constexpr real_t inv_cs2 = 1.0 / cs2;
    inline static constexpr real_t inv_cs4 = 1.0 / cs4;
    inline static constexpr real_t inv_cs6 = 1.0 / cs6;
    inline static constexpr real_t inv_cs8 = 1.0 / cs8;

    // ========================================================
    // Discrete velocities
    //
    // Opposite velocities are stored consecutively.
    // ========================================================

    inline static constexpr int cxTable[Q] =
    {
         0,
         1, -1,  0,  0,
         1, -1,  1, -1,
         2, -2,  0,  0,
         2, -2,  2, -2,  1, -1,  1, -1,
         2, -2,  2, -2,
         3, -3,  0,  0,
         3, -3,  3, -3,  1, -1,  1, -1
    };

    inline static constexpr int cyTable[Q] =
    {
         0,
         0,  0,  1, -1,
         1, -1, -1,  1,
         0,  0,  2, -2,
         1, -1, -1,  1,  2, -2, -2,  2,
         2, -2, -2,  2,
         0,  0,  3, -3,
         1, -1, -1,  1,  3, -3, -3,  3
    };

    // ========================================================
    // Opposite directions
    // ========================================================

    inline static constexpr label_t oppositeTable[Q] =
    {
         0,
         2,  1,  4,  3,
         6,  5,  8,  7,
        10,  9, 12, 11,
        14, 13, 16, 15, 18, 17, 20, 19,
        22, 21, 24, 23,
        26, 25, 28, 27,
        30, 29, 32, 31, 34, 33, 36, 35
    };

    // ========================================================
    // Quadrature weights
    //
    // One weight per discrete velocity.
    // The weight depends on cx^2 + cy^2.
    // ========================================================

    D2V37_HD [[nodiscard]]
    static constexpr real_t weightFromC2(const int c2) noexcept
    {
        return
            c2 == 0  ? 0.233150669132352502286506704067 :
            c2 == 1  ? 0.107306091542219002412464287183 :
            c2 == 2  ? 0.0576678598887948820300692153933 :
            c2 == 4  ? 0.0142082161584507502646989423441 :
            c2 == 5  ? 0.00535304900051377523273150166219 :
            c2 == 8  ? 0.00101193759267357547541090850664 :
            c2 == 9  ? 0.000245301027757717345465916643268 :
            c2 == 10 ? 0.000283414252994198217400525294195 :
            0.0;
    }

    // ========================================================
    // Compile-time velocity accessors
    // ========================================================

    template <label_t I>
    D2V37_HD [[nodiscard]]
    static constexpr int cx() noexcept
    {
        static_assert(I < Q);
        return cxTable[I];
    }

    template <label_t I>
    D2V37_HD [[nodiscard]]
    static constexpr int cy() noexcept
    {
        static_assert(I < Q);
        return cyTable[I];
    }

    template <label_t I>
    D2V37_HD [[nodiscard]]
    static constexpr int cz() noexcept
    {
        static_assert(I < Q);
        return 0;
    }

    // ========================================================
    // Squared discrete velocity
    // ========================================================

    template <label_t I>
    D2V37_HD [[nodiscard]]
    static constexpr int c2() noexcept
    {
        return cx<I>() * cx<I>() + cy<I>() * cy<I>();
    }

    // ========================================================
    // Compile-time quadrature weight
    // ========================================================

    template <label_t I>
    D2V37_HD [[nodiscard]]
    static constexpr real_t w() noexcept
    {
        static_assert(I < Q);
        return weightFromC2(c2<I>());
    }

    // ========================================================
    // Compile-time opposite direction
    // ========================================================

    template <label_t I>
    D2V37_HD [[nodiscard]]
    static constexpr label_t opp() noexcept
    {
        static_assert(I < Q);
        return oppositeTable[I];
    }

    // ========================================================
    // Validate opposite directions
    // ========================================================

    [[nodiscard]]
    static constexpr bool checkOpposites() noexcept
    {
        for (label_t i = 0; i < Q; ++i)
        {
            const label_t j = oppositeTable[i];

            if (j >= Q ||
                oppositeTable[j] != i ||
                cxTable[i] != -cxTable[j] ||
                cyTable[i] != -cyTable[j])
                return false;
        }

        return true;
    }

    // ========================================================
    // Validate velocity shells
    // ========================================================

    [[nodiscard]]
    static constexpr bool checkShells() noexcept
    {
        for (label_t i = 0; i < Q; ++i)
        {
            const int s =
                cxTable[i] * cxTable[i] +
                cyTable[i] * cyTable[i];

            if (!(s == 0 || s == 1 || s == 2 ||
                  s == 4 || s == 5 || s == 8 ||
                  s == 9 || s == 10))
                return false;
        }

        return true;
    }

    // ========================================================
    // Compile-time raw quadrature moment
    //
    // M(PX,PY) = sum_i wi * cx_i^PX * cy_i^PY
    // ========================================================

    template <int PX, int PY>
    [[nodiscard]]
    static constexpr real_t moment() noexcept
    {
        real_t sum = 0.0;

        for (label_t i = 0; i < Q; ++i)
        {
            const int x = cxTable[i];
            const int y = cyTable[i];

            real_t xPower = 1.0;
            real_t yPower = 1.0;

            for (int k = 0; k < PX; ++k)
                xPower *= x;

            for (int k = 0; k < PY; ++k)
                yPower *= y;

            sum += weightFromC2(x * x + y * y) *
                   xPower * yPower;
        }

        return sum;
    }

    // ========================================================
    // Compile-time floating-point comparison
    // ========================================================

    [[nodiscard]]
    static constexpr bool near(
        const real_t value,
        const real_t target) noexcept
    {
        return (value > target ? value - target : target - value)
               < 1.0e-11;
    }
};

// ============================================================
// Compile-time validation
// ============================================================

static_assert(D2V37::checkOpposites(),
              "D2V37 opposite directions are invalid.");

static_assert(D2V37::checkShells(),
              "D2V37 contains an invalid velocity shell.");

// Zeroth and odd moments
static_assert(D2V37::near(D2V37::moment<0, 0>(), 1.0));
static_assert(D2V37::near(D2V37::moment<1, 0>(), 0.0));
static_assert(D2V37::near(D2V37::moment<0, 1>(), 0.0));

// Second-order isotropy
static_assert(D2V37::near(D2V37::moment<2, 0>(), D2V37::cs2));
static_assert(D2V37::near(D2V37::moment<0, 2>(), D2V37::cs2));
static_assert(D2V37::near(D2V37::moment<1, 1>(), 0.0));

// Fourth-order isotropy
static_assert(D2V37::near(D2V37::moment<4, 0>(), 3.0 * D2V37::cs4));
static_assert(D2V37::near(D2V37::moment<2, 2>(), D2V37::cs4));

// Sixth-order isotropy
static_assert(D2V37::near(D2V37::moment<6, 0>(), 15.0 * D2V37::cs6));
static_assert(D2V37::near(D2V37::moment<4, 2>(), 3.0 * D2V37::cs6));

// Eighth-order isotropy
static_assert(D2V37::near(D2V37::moment<8, 0>(), 105.0 * D2V37::cs8));
static_assert(D2V37::near(D2V37::moment<6, 2>(), 15.0 * D2V37::cs8));
static_assert(D2V37::near(D2V37::moment<4, 4>(), 9.0 * D2V37::cs8));

#undef D2V37_HD

#endif // D2V37_CUH


#ifndef CONSTEXPR_FOR_CUH
#define CONSTEXPR_FOR_CUH

#include <type_traits>

#include "types.cuh"


template <label_t Begin, label_t End, typename Function>
__host__ __device__ __forceinline__
constexpr void constexprFor(Function &&function) noexcept
{
    if constexpr (Begin < End)
    {
        function(std::integral_constant<label_t, Begin>{});

        constexprFor<Begin + 1, End>(static_cast<Function&&>(function));
    }
}

#endif // CONSTEXPR_FOR_CUH

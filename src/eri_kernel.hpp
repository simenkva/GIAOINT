#pragma once

#include "giao_integrals/eri.hpp"

namespace giao::detail {

// Internal entry point also used to cross-check exact-zero dispatch in tests.
// Pair data must match the supplied primitives and field.
struct EriKernel {
    static bool use_real_path(const MagneticField& field) noexcept {
        return field.B.x == 0.0 && field.B.y == 0.0 && field.B.z == 0.0;
    }

    static Complex compute(const PrimitiveGaussian& a,
                           const PrimitiveGaussian& b,
                           const PrimitiveGaussian& c,
                           const PrimitiveGaussian& d,
                           const GaussianPairData& pair_ab,
                           const GaussianPairData& pair_cd,
                           bool real_zero_field,
                           EriWorkspace& workspace);
};

}  // namespace giao::detail

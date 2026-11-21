#pragma once

#include "giao_integrals/eri.hpp"
#include <array>
#include <span>

namespace giao::detail {

void cholesky_diagonal(const Basis& basis,
    std::span<const std::array<std::size_t,2>> pairs,
    const MagneticField& field,std::span<double> output,std::size_t threads);
void cholesky_columns(const Basis& basis,
    std::span<const std::array<std::size_t,2>> pivots,
    const MagneticField& field,std::span<double> output,std::size_t threads);
void cholesky_columns(const Basis& basis,
    std::span<const std::array<std::size_t,2>> pivots,
    const MagneticField& field,std::span<Complex> output,std::size_t threads);

} // namespace giao::detail

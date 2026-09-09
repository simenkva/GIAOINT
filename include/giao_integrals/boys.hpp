#pragma once

#include "giao_integrals/types.hpp"

#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace giao {

enum class BoysScaling {
    unscaled,
    exp_z,
};

enum class BoysRegion {
    power_series,
    adaptive_quadrature,
    scaled_quadrature,
    positive_asymptotic,
};

struct BoysDiagnostics {
    BoysRegion region{};
    BoysScaling scaling{};
    double estimated_absolute_error{};
    std::size_t quadrature_segments{};
};

class BoysNumericalError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

[[nodiscard]] BoysDiagnostics compute_boys(
    Complex argument, std::span<Complex> values,
    BoysScaling scaling = BoysScaling::unscaled);

[[nodiscard]] std::vector<Complex> boys_values(
    Complex argument, std::size_t maximum_order,
    BoysScaling scaling = BoysScaling::unscaled);

}  // namespace giao

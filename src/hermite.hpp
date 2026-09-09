#pragma once

#include <cstdint>
#include <vector>

namespace giao::detail {

void build_hermite_coefficients(std::uint16_t angular_bra,
                                std::uint16_t angular_ket,
                                double center_bra, double center_ket,
                                double product_center, double exponent,
                                std::vector<double>& current,
                                std::vector<double>& next);

}  // namespace giao::detail

#include "hermite.hpp"

#include <algorithm>
#include <cstddef>

namespace giao::detail {

void build_hermite_coefficients(std::uint16_t angular_bra,
                                std::uint16_t angular_ket,
                                double center_bra, double center_ket,
                                double product_center, double exponent,
                                std::vector<double>& current,
                                std::vector<double>& next) {
    const auto maximum_order = static_cast<std::size_t>(angular_bra) +
                               static_cast<std::size_t>(angular_ket);
    std::fill_n(current.begin(), maximum_order + 2U, 0.0);
    current[0] = 1.0;
    std::size_t active_order = 0;
    const double inverse_two_p = 0.5 / exponent;

    const auto raise = [&](double displacement) {
        std::fill_n(next.begin(), maximum_order + 2U, 0.0);
        for (std::size_t t = 0; t <= active_order + 1U; ++t) {
            const double lower = t == 0 ? 0.0 : current[t - 1U];
            const double same = t <= active_order ? current[t] : 0.0;
            const double upper = t < active_order ? current[t + 1U] : 0.0;
            next[t] = inverse_two_p * lower + displacement * same +
                      static_cast<double>(t + 1U) * upper;
        }
        ++active_order;
        current.swap(next);
    };

    for (std::uint16_t index = 0; index < angular_bra; ++index) {
        raise(product_center - center_bra);
    }
    for (std::uint16_t index = 0; index < angular_ket; ++index) {
        raise(product_center - center_ket);
    }
}

}  // namespace giao::detail

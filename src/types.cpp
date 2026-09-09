#include "giao_integrals/types.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>

namespace giao {
namespace {

void validate_finite(Vec3 value, const char* name) {
    if (!std::isfinite(value.x) || !std::isfinite(value.y) ||
        !std::isfinite(value.z)) {
        throw std::invalid_argument(std::string(name) + " values must be finite");
    }
}

double log_odd_double_factorial(std::uint16_t angular) {
    if (angular == 0) {
        return 0.0;
    }
    const auto n = static_cast<double>(angular);
    return n * std::log(2.0) + std::lgamma(n + 0.5) -
           0.5 * std::log(std::numbers::pi);
}

double same_center_primitive_overlap(double alpha, double beta,
                                     std::uint16_t angular_momentum) {
    const auto angular = CartesianExponent{angular_momentum, 0, 0};
    const double normalization =
        primitive_normalization(alpha, angular) *
        primitive_normalization(beta, angular);
    const double p = alpha + beta;
    const double l = static_cast<double>(angular_momentum);
    const double log_x_moment =
        std::lgamma(l + 0.5) - (l + 0.5) * std::log(p);
    return normalization * std::exp(log_x_moment) * std::numbers::pi / p;
}

}  // namespace

PrimitiveGaussian::PrimitiveGaussian(double exponent_in, Vec3 center_in,
                                     CartesianExponent angular_in,
                                     double coefficient_in,
                                     bool normalized_in)
    : exponent(exponent_in), coefficient(coefficient_in), center(center_in),
      angular(angular_in), normalized(normalized_in) {
    if (!std::isfinite(exponent) || exponent <= 0.0) {
        throw std::invalid_argument("Gaussian exponent must be finite and positive");
    }
    if (!std::isfinite(coefficient)) {
        throw std::invalid_argument("primitive coefficient must be finite");
    }
    validate_finite(center, "center");
}

double primitive_normalization(double exponent, CartesianExponent angular) {
    if (!std::isfinite(exponent) || exponent <= 0.0) {
        throw std::invalid_argument("Gaussian exponent must be finite and positive");
    }
    const double total = static_cast<double>(angular.total());
    const double log_denominator = log_odd_double_factorial(angular.x) +
                                   log_odd_double_factorial(angular.y) +
                                   log_odd_double_factorial(angular.z);
    const double log_normalization =
        0.75 * std::log(2.0 * exponent / std::numbers::pi) +
        0.5 * (total * std::log(4.0 * exponent) - log_denominator);
    return std::exp(log_normalization);
}

std::vector<CartesianExponent> cartesian_components(
    std::uint16_t total_angular_momentum) {
    const auto count =
        (static_cast<std::size_t>(total_angular_momentum) + 1U) *
        (static_cast<std::size_t>(total_angular_momentum) + 2U) / 2U;
    std::vector<CartesianExponent> result;
    result.reserve(count);
    for (int x = static_cast<int>(total_angular_momentum); x >= 0; --x) {
        const int remaining = static_cast<int>(total_angular_momentum) - x;
        for (int y = remaining; y >= 0; --y) {
            result.push_back(CartesianExponent{
                static_cast<std::uint16_t>(x), static_cast<std::uint16_t>(y),
                static_cast<std::uint16_t>(remaining - y)});
        }
    }
    return result;
}

Shell::Shell(Vec3 center, std::uint16_t angular_momentum,
             std::vector<double> exponents, std::vector<double> coefficients,
             std::size_t contraction_count,
             ContractionNormalization normalization)
    : center_(center), angular_momentum_(angular_momentum),
      exponents_(std::move(exponents)), coefficients_(std::move(coefficients)),
      contraction_count_(contraction_count), normalization_(normalization),
      components_(cartesian_components(angular_momentum)) {
    validate_finite(center_, "center");
    if (exponents_.empty()) {
        throw std::invalid_argument("a shell requires at least one primitive");
    }
    if (contraction_count_ == 0) {
        throw std::invalid_argument("a shell requires at least one contraction");
    }
    if (contraction_count_ >
        std::numeric_limits<std::size_t>::max() / exponents_.size() ||
        coefficients_.size() != contraction_count_ * exponents_.size()) {
        throw std::invalid_argument(
            "coefficient data must have contraction_count * primitive_count values");
    }
    for (double exponent : exponents_) {
        if (!std::isfinite(exponent) || exponent <= 0.0) {
            throw std::invalid_argument(
                "shell exponents must be finite and positive");
        }
    }
    for (double coefficient_value : coefficients_) {
        if (!std::isfinite(coefficient_value)) {
            throw std::invalid_argument("shell coefficients must be finite");
        }
    }

    for (std::size_t contraction = 0; contraction < contraction_count_;
         ++contraction) {
        bool any_nonzero = false;
        for (std::size_t primitive = 0; primitive < primitive_count(); ++primitive) {
            any_nonzero = any_nonzero || coefficient(contraction, primitive) != 0.0;
        }
        if (!any_nonzero) {
            throw std::invalid_argument(
                "a shell contraction cannot have all-zero coefficients");
        }

        if (normalization_ == ContractionNormalization::normalize) {
            double norm_squared = 0.0;
            for (std::size_t i = 0; i < primitive_count(); ++i) {
                for (std::size_t j = 0; j < primitive_count(); ++j) {
                    norm_squared += coefficient(contraction, i) *
                                    coefficient(contraction, j) *
                                    same_center_primitive_overlap(
                                        exponents_[i], exponents_[j],
                                        angular_momentum_);
                }
            }
            if (!std::isfinite(norm_squared) || norm_squared <= 0.0) {
                throw std::invalid_argument(
                    "contraction has a non-positive or non-finite norm");
            }
            const double scale = 1.0 / std::sqrt(norm_squared);
            for (std::size_t primitive = 0; primitive < primitive_count();
                 ++primitive) {
                coefficients_[contraction * primitive_count() + primitive] *=
                    scale;
            }
        }
    }
}

Basis::Basis(std::vector<Shell> shells) : shells_(std::move(shells)) {
    ao_offsets_.reserve(shells_.size());
    for (const auto& shell : shells_) {
        ao_offsets_.push_back(ao_count_);
        if (shell.ao_count() >
            std::numeric_limits<std::size_t>::max() - ao_count_) {
            throw std::overflow_error("basis AO count overflow");
        }
        ao_count_ += shell.ao_count();
    }
}

MagneticField::MagneticField(Vec3 field, Vec3 origin)
    : B(field), gauge_origin(origin) {
    validate_finite(B, "B");
    validate_finite(gauge_origin, "gauge_origin");
}

Vec3 MagneticField::london_wave_vector(Vec3 center) const noexcept {
    const Vec3 displacement{center.x - gauge_origin.x,
                            center.y - gauge_origin.y,
                            center.z - gauge_origin.z};
    return Vec3{0.5 * (B.y * displacement.z - B.z * displacement.y),
                0.5 * (B.z * displacement.x - B.x * displacement.z),
                0.5 * (B.x * displacement.y - B.y * displacement.x)};
}

Nucleus::Nucleus(double charge_in, Vec3 center_in)
    : charge(charge_in), center(center_in) {
    if (!std::isfinite(charge) || charge <= 0.0) {
        throw std::invalid_argument("nuclear charge must be finite and positive");
    }
    validate_finite(center, "nuclear center");
}

}  // namespace giao

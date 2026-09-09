#include "giao_integrals/boys.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <sstream>
#include <utility>
#include <vector>

namespace giao {
namespace {

constexpr std::size_t maximum_supported_order = 32;
constexpr double direct_argument_limit = 160.0;
constexpr double series_limit = 0.75;
constexpr double absolute_tolerance = 2.0e-15;
constexpr double relative_tolerance = 7.5e-14;

struct QuadratureRule {
    std::vector<double> nodes;
    std::vector<double> weights;
};

const QuadratureRule& gauss_legendre_rule(std::size_t order) {
    static const QuadratureRule rule_16 = [] {
        QuadratureRule result;
        constexpr std::size_t n = 16;
        result.nodes.resize(n);
        result.weights.resize(n);
        for (std::size_t i = 0; i < n / 2; ++i) {
            double root = std::cos(std::numbers::pi *
                                   (static_cast<double>(i) + 0.75) /
                                   (static_cast<double>(n) + 0.5));
            double derivative = 0.0;
            for (int iteration = 0; iteration < 20; ++iteration) {
                double previous = 1.0;
                double current = root;
                for (std::size_t degree = 2; degree <= n; ++degree) {
                    const double next =
                        ((2.0 * static_cast<double>(degree) - 1.0) * root *
                             current -
                         (static_cast<double>(degree) - 1.0) * previous) /
                        static_cast<double>(degree);
                    previous = current;
                    current = next;
                }
                derivative = static_cast<double>(n) *
                             (root * current - previous) /
                             (root * root - 1.0);
                const double update = current / derivative;
                root -= update;
                if (std::abs(update) < 4.0 * std::numeric_limits<double>::epsilon()) {
                    break;
                }
            }
            const double weight = 2.0 / ((1.0 - root * root) *
                                         derivative * derivative);
            result.nodes[i] = -root;
            result.nodes[n - 1U - i] = root;
            result.weights[i] = weight;
            result.weights[n - 1U - i] = weight;
        }
        return result;
    }();
    static const QuadratureRule rule_32 = [] {
        QuadratureRule result;
        constexpr std::size_t n = 32;
        result.nodes.resize(n);
        result.weights.resize(n);
        for (std::size_t i = 0; i < n / 2; ++i) {
            double root = std::cos(std::numbers::pi *
                                   (static_cast<double>(i) + 0.75) /
                                   (static_cast<double>(n) + 0.5));
            double derivative = 0.0;
            for (int iteration = 0; iteration < 20; ++iteration) {
                double previous = 1.0;
                double current = root;
                for (std::size_t degree = 2; degree <= n; ++degree) {
                    const double next =
                        ((2.0 * static_cast<double>(degree) - 1.0) * root *
                             current -
                         (static_cast<double>(degree) - 1.0) * previous) /
                        static_cast<double>(degree);
                    previous = current;
                    current = next;
                }
                derivative = static_cast<double>(n) *
                             (root * current - previous) /
                             (root * root - 1.0);
                const double update = current / derivative;
                root -= update;
                if (std::abs(update) < 4.0 * std::numeric_limits<double>::epsilon()) {
                    break;
                }
            }
            const double weight = 2.0 / ((1.0 - root * root) *
                                         derivative * derivative);
            result.nodes[i] = -root;
            result.nodes[n - 1U - i] = root;
            result.weights[i] = weight;
            result.weights[n - 1U - i] = weight;
        }
        return result;
    }();
    return order == 16 ? rule_16 : rule_32;
}

void quadrature(Complex argument, BoysScaling scaling, std::size_t segments,
                const QuadratureRule& rule, std::span<Complex> values) {
    std::fill(values.begin(), values.end(), Complex{});
    const double inverse_segments = 1.0 / static_cast<double>(segments);
    for (std::size_t segment = 0; segment < segments; ++segment) {
        const double midpoint =
            (static_cast<double>(segment) + 0.5) * inverse_segments;
        const double half_width = 0.5 * inverse_segments;
        for (std::size_t point = 0; point < rule.nodes.size(); ++point) {
            const double t = midpoint + half_width * rule.nodes[point];
            const double t_squared = t * t;
            const Complex exponent =
                scaling == BoysScaling::unscaled
                    ? -argument * t_squared
                    : argument * (1.0 - t_squared);
            const Complex common =
                half_width * rule.weights[point] * std::exp(exponent);
            double power = 1.0;
            for (auto& value : values) {
                value += common * power;
                power *= t_squared;
            }
        }
    }
}

BoysDiagnostics power_series(Complex argument, BoysScaling scaling,
                             std::span<Complex> values) {
    double worst_last_term = 0.0;
    for (std::size_t n = 0; n < values.size(); ++n) {
        Complex power_over_factorial{1.0, 0.0};
        Complex sum{};
        Complex compensation{};
        double last_term = 0.0;
        for (std::size_t k = 0; k < 256; ++k) {
            const Complex term = power_over_factorial /
                                 static_cast<double>(2U * n + 2U * k + 1U);
            const Complex corrected = term - compensation;
            const Complex next = sum + corrected;
            compensation = (next - sum) - corrected;
            sum = next;
            last_term = std::abs(term);
            if (k > 2U &&
                last_term <= absolute_tolerance +
                                 relative_tolerance * std::abs(sum)) {
                break;
            }
            power_over_factorial *=
                -argument / static_cast<double>(k + 1U);
            if (k == 255U) {
                throw BoysNumericalError("complex Boys power series did not converge");
            }
        }
        values[n] = scaling == BoysScaling::unscaled ? sum
                                                     : std::exp(argument) * sum;
        worst_last_term = std::max(worst_last_term, last_term);
    }
    return {BoysRegion::power_series, scaling, worst_last_term, 0U};
}

BoysDiagnostics adaptive_quadrature(Complex argument, BoysScaling scaling,
                                    std::span<Complex> values,
                                    BoysRegion region) {
    const double difficulty = std::max(std::abs(argument.real()),
                                       std::abs(argument.imag()));
    std::size_t segments = std::max<std::size_t>(
        1U, static_cast<std::size_t>(std::ceil(difficulty / 8.0)));
    std::array<Complex, maximum_supported_order + 1U> low{};
    std::array<Complex, maximum_supported_order + 1U> high{};
    double worst_error = 0.0;
    for (;;) {
        auto low_values = std::span<Complex>(low.data(), values.size());
        auto high_values = std::span<Complex>(high.data(), values.size());
        quadrature(argument, scaling, segments, gauss_legendre_rule(16),
                   low_values);
        quadrature(argument, scaling, segments, gauss_legendre_rule(32),
                   high_values);
        bool converged = true;
        worst_error = 0.0;
        for (std::size_t n = 0; n < values.size(); ++n) {
            const double error = std::abs(high[n] - low[n]);
            worst_error = std::max(worst_error, error);
            converged = converged &&
                        error <= absolute_tolerance +
                                     relative_tolerance * std::abs(high[n]);
        }
        if (converged) {
            std::copy_n(high.begin(), values.size(), values.begin());
            return {region, scaling, worst_error, segments};
        }
        if (segments >= 4096U) {
            throw BoysNumericalError(
                "adaptive complex Boys quadrature did not reach tolerance");
        }
        segments *= 2U;
    }
}

BoysDiagnostics positive_asymptotic(Complex argument, BoysScaling scaling,
                                    std::span<Complex> values) {
    const Complex square_root = std::sqrt(argument);
    Complex inverse_power = 1.0 / square_root;
    double gamma = std::sqrt(std::numbers::pi);
    for (std::size_t n = 0; n < values.size(); ++n) {
        Complex value = 0.5 * gamma * inverse_power;
        if (scaling == BoysScaling::exp_z) {
            value *= std::exp(argument);
        }
        if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) {
            throw BoysNumericalError(
                "requested Boys scaling is not representable for this argument");
        }
        values[n] = value;
        gamma *= static_cast<double>(n) + 0.5;
        inverse_power /= argument;
    }
    const double tail_bound =
        std::abs(std::exp(-argument)) /
        std::max(1.0, 2.0 * std::abs(argument));
    return {BoysRegion::positive_asymptotic, scaling, tail_bound, 0U};
}

[[noreturn]] void unsupported(Complex argument, std::size_t maximum_order) {
    std::ostringstream message;
    message << "complex Boys argument " << argument << " at order "
            << maximum_order
            << " is outside the verified direct domain |z| <= "
            << direct_argument_limit
            << " and the conservative large-positive asymptotic sector";
    throw BoysNumericalError(message.str());
}

}  // namespace

BoysDiagnostics compute_boys(Complex argument, std::span<Complex> values,
                             BoysScaling scaling) {
    if (values.empty()) {
        throw std::invalid_argument("Boys output must contain at least F_0");
    }
    const std::size_t maximum_order = values.size() - 1U;
    if (maximum_order > maximum_supported_order) {
        throw std::invalid_argument("complex Boys supports orders through 32");
    }
    if (!std::isfinite(argument.real()) || !std::isfinite(argument.imag())) {
        throw std::invalid_argument("complex Boys argument must be finite");
    }
    const double magnitude = std::abs(argument);
    if (magnitude <= series_limit) {
        return power_series(argument, scaling, values);
    }

    const double asymptotic_threshold =
        2.0 * static_cast<double>(maximum_order) + 60.0;
    if (argument.real() >= asymptotic_threshold &&
        std::abs(argument.imag()) <= 0.5 * argument.real()) {
        return positive_asymptotic(argument, scaling, values);
    }
    if (magnitude > direct_argument_limit) {
        unsupported(argument, maximum_order);
    }

    if (argument.real() < 0.0) {
        if (scaling == BoysScaling::exp_z) {
            return adaptive_quadrature(argument, scaling, values,
                                       BoysRegion::scaled_quadrature);
        }
        auto diagnostics = adaptive_quadrature(
            argument, BoysScaling::exp_z, values,
            BoysRegion::scaled_quadrature);
        const Complex factor = std::exp(-argument);
        for (auto& value : values) {
            value *= factor;
            if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) {
                throw BoysNumericalError(
                    "unscaled Boys value is not representable for this argument");
            }
        }
        diagnostics.scaling = scaling;
        diagnostics.estimated_absolute_error *= std::abs(factor);
        return diagnostics;
    }
    return adaptive_quadrature(argument, scaling, values,
                               BoysRegion::adaptive_quadrature);
}

std::vector<Complex> boys_values(Complex argument, std::size_t maximum_order,
                                 BoysScaling scaling) {
    std::vector<Complex> values(maximum_order + 1U);
    [[maybe_unused]] const auto diagnostics =
        compute_boys(argument, values, scaling);
    return values;
}

}  // namespace giao

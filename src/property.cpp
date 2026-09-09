#include "giao_integrals/property.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace giao {
namespace {

std::size_t axis_index(Axis axis) {
    const auto index = static_cast<std::size_t>(axis);
    if (index > 2U) {
        throw std::invalid_argument("axis must be x, y, or z");
    }
    return index;
}

double integer_power(double base, std::uint16_t exponent) {
    double result = 1.0;
    double factor = base;
    auto remaining = exponent;
    while (remaining != 0U) {
        if ((remaining & 1U) != 0U) {
            result *= factor;
        }
        factor *= factor;
        remaining = static_cast<std::uint16_t>(remaining >> 1U);
    }
    return result;
}

double binomial(std::uint16_t n, std::uint16_t k) {
    if (k > n) {
        return 0.0;
    }
    auto reduced_k = std::min(k, static_cast<std::uint16_t>(n - k));
    double result = 1.0;
    for (std::uint32_t index = 1; index <= reduced_k; ++index) {
        result *= static_cast<double>(n - reduced_k + index) /
                  static_cast<double>(index);
    }
    return result;
}

CartesianExponent shifted(CartesianExponent angular, std::size_t axis,
                           int amount) {
    auto values = std::array<std::uint16_t, 3>{angular.x, angular.y, angular.z};
    const int value = static_cast<int>(values[axis]) + amount;
    if (value < 0 || value > std::numeric_limits<std::uint16_t>::max()) {
        throw std::overflow_error("shifted angular momentum is out of range");
    }
    values[axis] = static_cast<std::uint16_t>(value);
    return {values[0], values[1], values[2]};
}

PrimitiveGaussian raw_primitive(const PrimitiveGaussian& primitive) {
    const double normalization =
        primitive.normalized
            ? primitive_normalization(primitive.exponent, primitive.angular)
            : 1.0;
    return PrimitiveGaussian(primitive.exponent, primitive.center,
                             primitive.angular,
                             primitive.coefficient * normalization, false);
}

Complex shifted_ket_overlap(const PrimitiveGaussian& raw_bra,
                            const PrimitiveGaussian& raw_ket,
                            CartesianExponent angular_ket,
                            const MagneticField& field,
                            IntegralWorkspace& workspace) {
    const PrimitiveGaussian shifted_ket(
        raw_ket.exponent, raw_ket.center, angular_ket, raw_ket.coefficient, false);
    return primitive_overlap(raw_bra, shifted_ket, field, workspace);
}

Complex raw_moment(const PrimitiveGaussian& raw_bra,
                   const PrimitiveGaussian& raw_ket,
                   const CartesianMoment& moment, const MagneticField& field,
                   IntegralWorkspace& workspace) {
    Complex result{};
    for (std::uint32_t x = 0; x <= moment.powers.x; ++x) {
        const auto x_power = static_cast<std::uint16_t>(x);
        const double coefficient_x =
            binomial(moment.powers.x, x_power) *
            integer_power(raw_ket.center.x - moment.origin.x,
                          static_cast<std::uint16_t>(moment.powers.x - x_power));
        for (std::uint32_t y = 0; y <= moment.powers.y; ++y) {
            const auto y_power = static_cast<std::uint16_t>(y);
            const double coefficient_y =
                binomial(moment.powers.y, y_power) *
                integer_power(raw_ket.center.y - moment.origin.y,
                              static_cast<std::uint16_t>(moment.powers.y - y_power));
            for (std::uint32_t z = 0; z <= moment.powers.z; ++z) {
                const auto z_power = static_cast<std::uint16_t>(z);
                const double coefficient_z =
                    binomial(moment.powers.z, z_power) *
                    integer_power(raw_ket.center.z - moment.origin.z,
                                  static_cast<std::uint16_t>(moment.powers.z - z_power));
                CartesianExponent angular =
                    shifted(raw_ket.angular, 0, static_cast<int>(x));
                angular = shifted(angular, 1, static_cast<int>(y));
                angular = shifted(angular, 2, static_cast<int>(z));
                result += coefficient_x * coefficient_y * coefficient_z *
                          shifted_ket_overlap(raw_bra, raw_ket, angular, field,
                                              workspace);
            }
        }
    }
    return result;
}

Complex raw_gradient(const PrimitiveGaussian& raw_bra,
                     const PrimitiveGaussian& raw_ket, std::size_t axis,
                     const MagneticField& field,
                     IntegralWorkspace& workspace) {
    const auto angular = raw_ket.angular[axis];
    Complex result{};
    if (angular != 0U) {
        result += static_cast<double>(angular) *
                  shifted_ket_overlap(raw_bra, raw_ket,
                                      shifted(raw_ket.angular, axis, -1), field,
                                      workspace);
    }
    result -= 2.0 * raw_ket.exponent *
              shifted_ket_overlap(raw_bra, raw_ket,
                                  shifted(raw_ket.angular, axis, 1), field,
                                  workspace);
    const Vec3 wave_vector = field.london_wave_vector(raw_ket.center);
    result += Complex{0.0, -wave_vector[axis]} *
              shifted_ket_overlap(raw_bra, raw_ket, raw_ket.angular, field,
                                  workspace);
    return result;
}

Complex raw_momentum_moment(const PrimitiveGaussian& raw_bra,
                            const PrimitiveGaussian& raw_ket,
                            std::size_t coordinate_axis,
                            std::size_t momentum_axis,
                            const MagneticField& field,
                            IntegralWorkspace& workspace) {
    auto powers = CartesianExponent{};
    if (coordinate_axis == 0U) {
        powers.x = 1;
    } else if (coordinate_axis == 1U) {
        powers.y = 1;
    } else {
        powers.z = 1;
    }
    const CartesianMoment coordinate(field.gauge_origin, powers);
    const Vec3 wave_vector = field.london_wave_vector(raw_ket.center);
    Complex derivative_moment{};
    const auto angular = raw_ket.angular[momentum_axis];
    if (angular != 0U) {
        const PrimitiveGaussian lower(
            raw_ket.exponent, raw_ket.center,
            shifted(raw_ket.angular, momentum_axis, -1), raw_ket.coefficient,
            false);
        derivative_moment +=
            static_cast<double>(angular) *
            raw_moment(raw_bra, lower, coordinate, field, workspace);
    }
    const PrimitiveGaussian upper(
        raw_ket.exponent, raw_ket.center,
        shifted(raw_ket.angular, momentum_axis, 1), raw_ket.coefficient, false);
    derivative_moment -= 2.0 * raw_ket.exponent *
                         raw_moment(raw_bra, upper, coordinate, field, workspace);
    const Complex base_moment =
        raw_moment(raw_bra, raw_ket, coordinate, field, workspace);
    return Complex{0.0, -1.0} * derivative_moment -
           wave_vector[momentum_axis] * base_moment;
}

template <typename Evaluator>
void compute_property(const Shell& a, const Shell& b, std::span<Complex> output,
                      IntegralWorkspace& workspace, std::size_t extra_order,
                      Evaluator&& evaluate) {
    if (output.size() != shell_pair_size(a, b)) {
        throw std::length_error("property output has an incorrect size");
    }
    std::fill(output.begin(), output.end(), Complex{});
    workspace.prepare_overlap(
        static_cast<std::size_t>(a.angular_momentum()) + b.angular_momentum() +
        extra_order);

    const auto component_count_a = a.cartesian_count();
    const auto component_count_b = b.cartesian_count();
    std::vector<double> normalizations_a(component_count_a * a.primitive_count());
    std::vector<double> normalizations_b(component_count_b * b.primitive_count());
    for (std::size_t component = 0; component < component_count_a; ++component) {
        for (std::size_t primitive = 0; primitive < a.primitive_count();
             ++primitive) {
            normalizations_a[component * a.primitive_count() + primitive] =
                primitive_normalization(a.exponents()[primitive],
                                        a.components()[component]);
        }
    }
    for (std::size_t component = 0; component < component_count_b; ++component) {
        for (std::size_t primitive = 0; primitive < b.primitive_count();
             ++primitive) {
            normalizations_b[component * b.primitive_count() + primitive] =
                primitive_normalization(b.exponents()[primitive],
                                        b.components()[component]);
        }
    }

    for (std::size_t primitive_a = 0; primitive_a < a.primitive_count();
         ++primitive_a) {
        for (std::size_t primitive_b = 0; primitive_b < b.primitive_count();
             ++primitive_b) {
            for (std::size_t component_a = 0; component_a < component_count_a;
                 ++component_a) {
                const PrimitiveGaussian bra(
                    a.exponents()[primitive_a], a.center(),
                    a.components()[component_a],
                    normalizations_a[component_a * a.primitive_count() +
                                     primitive_a],
                    false);
                for (std::size_t component_b = 0;
                     component_b < component_count_b; ++component_b) {
                    const PrimitiveGaussian ket(
                        b.exponents()[primitive_b], b.center(),
                        b.components()[component_b],
                        normalizations_b[component_b * b.primitive_count() +
                                         primitive_b],
                        false);
                    const Complex primitive_value = evaluate(bra, ket, workspace);
                    for (std::size_t contraction_a = 0;
                         contraction_a < a.contraction_count(); ++contraction_a) {
                        const std::size_t ao_a =
                            contraction_a * component_count_a + component_a;
                        const double coefficient_a =
                            a.coefficient(contraction_a, primitive_a);
                        for (std::size_t contraction_b = 0;
                             contraction_b < b.contraction_count();
                             ++contraction_b) {
                            const std::size_t ao_b =
                                contraction_b * component_count_b + component_b;
                            output[ao_a * b.ao_count() + ao_b] +=
                                coefficient_a *
                                b.coefficient(contraction_b, primitive_b) *
                                primitive_value;
                        }
                    }
                }
            }
        }
    }
}

template <typename ShellEvaluator>
void compute_property_matrix(const Basis& basis, std::span<Complex> output,
                             ShellEvaluator&& evaluate_shell) {
    const auto ao_count = basis.ao_count();
    if (ao_count != 0 &&
        ao_count > std::numeric_limits<std::size_t>::max() / ao_count) {
        throw std::length_error("basis property size overflow");
    }
    if (output.size() != ao_count * ao_count) {
        throw std::length_error("basis property output has an incorrect size");
    }
    std::fill(output.begin(), output.end(), Complex{});
    IntegralWorkspace workspace;
    std::vector<Complex> block;
    for (std::size_t shell_a = 0; shell_a < basis.shells().size(); ++shell_a) {
        const auto& a = basis.shells()[shell_a];
        for (std::size_t shell_b = 0; shell_b < basis.shells().size(); ++shell_b) {
            const auto& b = basis.shells()[shell_b];
            block.resize(shell_pair_size(a, b));
            evaluate_shell(a, b, block, workspace);
            for (std::size_t ao_a = 0; ao_a < a.ao_count(); ++ao_a) {
                for (std::size_t ao_b = 0; ao_b < b.ao_count(); ++ao_b) {
                    const auto global_a = basis.ao_offsets()[shell_a] + ao_a;
                    const auto global_b = basis.ao_offsets()[shell_b] + ao_b;
                    output[global_a * ao_count + global_b] =
                        block[ao_a * b.ao_count() + ao_b];
                }
            }
        }
    }
}

}  // namespace

CartesianMoment::CartesianMoment(Vec3 origin_in, CartesianExponent powers_in)
    : origin(origin_in), powers(powers_in) {
    if (!std::isfinite(origin.x) || !std::isfinite(origin.y) ||
        !std::isfinite(origin.z)) {
        throw std::invalid_argument("moment origin values must be finite");
    }
}

Complex primitive_moment(const PrimitiveGaussian& bra,
                         const PrimitiveGaussian& ket,
                         const CartesianMoment& moment,
                         const MagneticField& field,
                         IntegralWorkspace& workspace) {
    const auto total_order = static_cast<std::size_t>(bra.angular.total()) +
                             ket.angular.total() + moment.powers.total();
    workspace.prepare_overlap(total_order);
    return raw_moment(raw_primitive(bra), raw_primitive(ket), moment, field,
                      workspace);
}

Complex primitive_gradient(const PrimitiveGaussian& bra,
                           const PrimitiveGaussian& ket, Axis axis,
                           const MagneticField& field,
                           IntegralWorkspace& workspace) {
    workspace.prepare_overlap(static_cast<std::size_t>(bra.angular.total()) +
                              ket.angular.total() + 1U);
    return raw_gradient(raw_primitive(bra), raw_primitive(ket), axis_index(axis),
                        field, workspace);
}

Complex primitive_momentum(const PrimitiveGaussian& bra,
                           const PrimitiveGaussian& ket, Axis axis,
                           const MagneticField& field,
                           IntegralWorkspace& workspace) {
    return Complex{0.0, -1.0} *
           primitive_gradient(bra, ket, axis, field, workspace);
}

Complex primitive_kinetic(const PrimitiveGaussian& bra,
                          const PrimitiveGaussian& ket,
                          const MagneticField& field,
                          IntegralWorkspace& workspace) {
    workspace.prepare_overlap(static_cast<std::size_t>(bra.angular.total()) +
                              ket.angular.total() + 2U);
    const auto raw_bra = raw_primitive(bra);
    const auto raw_ket = raw_primitive(ket);
    const Vec3 wave_vector = field.london_wave_vector(raw_ket.center);
    Complex result{};
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const auto angular = raw_ket.angular[axis];
        if (angular >= 2U) {
            result -= 0.5 * static_cast<double>(angular) *
                      static_cast<double>(angular - 1U) *
                      shifted_ket_overlap(raw_bra, raw_ket,
                                          shifted(raw_ket.angular, axis, -2),
                                          field, workspace);
        }
        result += (raw_ket.exponent *
                       (2.0 * static_cast<double>(angular) + 1.0) +
                   0.5 * wave_vector[axis] * wave_vector[axis]) *
                  shifted_ket_overlap(raw_bra, raw_ket, raw_ket.angular, field,
                                      workspace);
        result -= 2.0 * raw_ket.exponent * raw_ket.exponent *
                  shifted_ket_overlap(raw_bra, raw_ket,
                                      shifted(raw_ket.angular, axis, 2), field,
                                      workspace);
        Complex first_derivative{};
        if (angular != 0U) {
            first_derivative +=
                static_cast<double>(angular) *
                shifted_ket_overlap(raw_bra, raw_ket,
                                    shifted(raw_ket.angular, axis, -1), field,
                                    workspace);
        }
        first_derivative -=
            2.0 * raw_ket.exponent *
            shifted_ket_overlap(raw_bra, raw_ket,
                                shifted(raw_ket.angular, axis, 1), field,
                                workspace);
        result += Complex{0.0, wave_vector[axis]} * first_derivative;
    }
    return result;
}

Complex primitive_magnetic_kinetic(const PrimitiveGaussian& bra,
                                   const PrimitiveGaussian& ket,
                                   const MagneticField& field,
                                   IntegralWorkspace& workspace) {
    workspace.prepare_overlap(static_cast<std::size_t>(bra.angular.total()) +
                              ket.angular.total() + 2U);
    const auto raw_bra = raw_primitive(bra);
    const auto raw_ket = raw_primitive(ket);
    Complex result = primitive_kinetic(raw_bra, raw_ket, field, workspace);

    const std::array<double, 3> magnetic_field{field.B.x, field.B.y, field.B.z};
    // 0.5 B . [(r-O) x p]
    result += 0.5 * magnetic_field[0] *
              (raw_momentum_moment(raw_bra, raw_ket, 1, 2, field, workspace) -
               raw_momentum_moment(raw_bra, raw_ket, 2, 1, field, workspace));
    result += 0.5 * magnetic_field[1] *
              (raw_momentum_moment(raw_bra, raw_ket, 2, 0, field, workspace) -
               raw_momentum_moment(raw_bra, raw_ket, 0, 2, field, workspace));
    result += 0.5 * magnetic_field[2] *
              (raw_momentum_moment(raw_bra, raw_ket, 0, 1, field, workspace) -
               raw_momentum_moment(raw_bra, raw_ket, 1, 0, field, workspace));

    const double field_squared = field.B.x * field.B.x + field.B.y * field.B.y +
                                 field.B.z * field.B.z;
    Complex radius_squared{};
    Complex projected_squared{};
    for (std::size_t axis = 0; axis < 3; ++axis) {
        auto powers = CartesianExponent{};
        if (axis == 0U) {
            powers.x = 2;
        } else if (axis == 1U) {
            powers.y = 2;
        } else {
            powers.z = 2;
        }
        radius_squared += raw_moment(
            raw_bra, raw_ket, CartesianMoment(field.gauge_origin, powers), field,
            workspace);
        projected_squared += magnetic_field[axis] * magnetic_field[axis] *
                             raw_moment(raw_bra, raw_ket,
                                        CartesianMoment(field.gauge_origin, powers),
                                        field, workspace);
    }
    for (std::size_t first = 0; first < 3; ++first) {
        for (std::size_t second = first + 1U; second < 3; ++second) {
            auto powers = CartesianExponent{};
            if (first == 0U || second == 0U) {
                powers.x = 1;
            }
            if (first == 1U || second == 1U) {
                powers.y = 1;
            }
            if (first == 2U || second == 2U) {
                powers.z = 1;
            }
            projected_squared +=
                2.0 * magnetic_field[first] * magnetic_field[second] *
                raw_moment(raw_bra, raw_ket,
                           CartesianMoment(field.gauge_origin, powers), field,
                           workspace);
        }
    }
    result += 0.125 *
              (field_squared * radius_squared - projected_squared);
    return result;
}

void compute_moment(const Shell& a, const Shell& b,
                    const CartesianMoment& moment,
                    const MagneticField& field, std::span<Complex> output,
                    IntegralWorkspace& workspace) {
    compute_property(a, b, output, workspace, moment.powers.total(),
                     [&](const auto& bra, const auto& ket, auto& scratch) {
                         return primitive_moment(bra, ket, moment, field, scratch);
                     });
}

void compute_gradient(const Shell& a, const Shell& b, Axis axis,
                      const MagneticField& field, std::span<Complex> output,
                      IntegralWorkspace& workspace) {
    static_cast<void>(axis_index(axis));
    compute_property(a, b, output, workspace, 1,
                     [&](const auto& bra, const auto& ket, auto& scratch) {
                         return primitive_gradient(bra, ket, axis, field, scratch);
                     });
}

void compute_momentum(const Shell& a, const Shell& b, Axis axis,
                      const MagneticField& field, std::span<Complex> output,
                      IntegralWorkspace& workspace) {
    static_cast<void>(axis_index(axis));
    compute_property(a, b, output, workspace, 1,
                     [&](const auto& bra, const auto& ket, auto& scratch) {
                         return primitive_momentum(bra, ket, axis, field, scratch);
                     });
}

void compute_kinetic(const Shell& a, const Shell& b,
                     const MagneticField& field, std::span<Complex> output,
                     IntegralWorkspace& workspace) {
    compute_property(a, b, output, workspace, 2,
                     [&](const auto& bra, const auto& ket, auto& scratch) {
                         return primitive_kinetic(bra, ket, field, scratch);
                     });
}

void compute_magnetic_kinetic(const Shell& a, const Shell& b,
                              const MagneticField& field,
                              std::span<Complex> output,
                              IntegralWorkspace& workspace) {
    compute_property(a, b, output, workspace, 2,
                     [&](const auto& bra, const auto& ket, auto& scratch) {
                         return primitive_magnetic_kinetic(bra, ket, field,
                                                           scratch);
                     });
}

void compute_moment_matrix(const Basis& basis, const CartesianMoment& moment,
                           const MagneticField& field,
                           std::span<Complex> output) {
    compute_property_matrix(
        basis, output, [&](const auto& a, const auto& b, auto& block, auto& scratch) {
            compute_moment(a, b, moment, field, block, scratch);
        });
}

void compute_gradient_matrix(const Basis& basis, Axis axis,
                             const MagneticField& field,
                             std::span<Complex> output) {
    compute_property_matrix(
        basis, output, [&](const auto& a, const auto& b, auto& block, auto& scratch) {
            compute_gradient(a, b, axis, field, block, scratch);
        });
}

void compute_momentum_matrix(const Basis& basis, Axis axis,
                             const MagneticField& field,
                             std::span<Complex> output) {
    compute_property_matrix(
        basis, output, [&](const auto& a, const auto& b, auto& block, auto& scratch) {
            compute_momentum(a, b, axis, field, block, scratch);
        });
}

void compute_kinetic_matrix(const Basis& basis, const MagneticField& field,
                            std::span<Complex> output) {
    compute_property_matrix(
        basis, output, [&](const auto& a, const auto& b, auto& block, auto& scratch) {
            compute_kinetic(a, b, field, block, scratch);
        });
}

void compute_magnetic_kinetic_matrix(const Basis& basis,
                                     const MagneticField& field,
                                     std::span<Complex> output) {
    compute_property_matrix(
        basis, output, [&](const auto& a, const auto& b, auto& block, auto& scratch) {
            compute_magnetic_kinetic(a, b, field, block, scratch);
        });
}

}  // namespace giao

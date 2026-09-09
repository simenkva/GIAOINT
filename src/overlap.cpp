#include "giao_integrals/overlap.hpp"

#include "hermite.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace giao {
namespace {

double dot(Vec3 left, Vec3 right) noexcept {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

double squared_distance(Vec3 left, Vec3 right) noexcept {
    const Vec3 difference{left.x - right.x, left.y - right.y,
                          left.z - right.z};
    return dot(difference, difference);
}

Complex one_dimensional_overlap(std::uint16_t angular_bra,
                                std::uint16_t angular_ket,
                                double center_bra, double center_ket,
                                double product_center, double exponent,
                                double pair_wave_vector,
                                std::vector<double>& current,
                                std::vector<double>& next) {
    const auto maximum_order = static_cast<std::size_t>(angular_bra) +
                               static_cast<std::size_t>(angular_ket);
    detail::build_hermite_coefficients(
        angular_bra, angular_ket, center_bra, center_ket, product_center,
        exponent, current, next);

    Complex result{};
    Complex fourier_power{1.0, 0.0};
    const Complex fourier_factor{0.0, -pair_wave_vector};
    for (std::size_t t = 0; t <= maximum_order; ++t) {
        result += current[t] * fourier_power;
        fourier_power *= fourier_factor;
    }
    return result;
}

}  // namespace

void IntegralWorkspace::prepare_overlap(std::size_t maximum_hermite_order,
                                        std::size_t normalization_count_a,
                                        std::size_t normalization_count_b) {
    const auto hermite_size = maximum_hermite_order + 2U;
    hermite_current_.resize(hermite_size);
    hermite_next_.resize(hermite_size);
    normalizations_a_.resize(normalization_count_a);
    normalizations_b_.resize(normalization_count_b);
}

GaussianPairData gaussian_product(const PrimitiveGaussian& bra,
                                  const PrimitiveGaussian& ket,
                                  const MagneticField& field) {
    const double exponent = bra.exponent + ket.exponent;
    const double reduced_exponent = bra.exponent * ket.exponent / exponent;
    const Vec3 product_center{
        (bra.exponent * bra.center.x + ket.exponent * ket.center.x) / exponent,
        (bra.exponent * bra.center.y + ket.exponent * ket.center.y) / exponent,
        (bra.exponent * bra.center.z + ket.exponent * ket.center.z) / exponent};
    const Vec3 wave_bra = field.london_wave_vector(bra.center);
    const Vec3 wave_ket = field.london_wave_vector(ket.center);
    const Vec3 pair_wave_vector{wave_ket.x - wave_bra.x,
                                wave_ket.y - wave_bra.y,
                                wave_ket.z - wave_bra.z};
    const double ordinary_prefactor = std::exp(
        -reduced_exponent * squared_distance(bra.center, ket.center));
    const double phase = -dot(pair_wave_vector, product_center);
    const double damping = -dot(pair_wave_vector, pair_wave_vector) /
                           (4.0 * exponent);
    const Complex london_prefactor = std::exp(Complex{damping, phase});

    return GaussianPairData{
        exponent,
        reduced_exponent,
        product_center,
        pair_wave_vector,
        {Complex{product_center.x, -0.5 * pair_wave_vector.x / exponent},
         Complex{product_center.y, -0.5 * pair_wave_vector.y / exponent},
         Complex{product_center.z, -0.5 * pair_wave_vector.z / exponent}},
        ordinary_prefactor,
        london_prefactor};
}

Complex primitive_overlap(const PrimitiveGaussian& bra,
                          const PrimitiveGaussian& ket,
                          const MagneticField& field,
                          IntegralWorkspace& workspace) {
    workspace.prepare_overlap(
        static_cast<std::size_t>(bra.angular.total() + ket.angular.total()));
    const auto pair = gaussian_product(bra, ket, field);
    Complex cartesian{1.0, 0.0};
    for (std::size_t axis = 0; axis < 3; ++axis) {
        cartesian *= one_dimensional_overlap(
            bra.angular[axis], ket.angular[axis], bra.center[axis],
            ket.center[axis], pair.product_center[axis], pair.exponent,
            pair.pair_wave_vector[axis], workspace.hermite_current_,
            workspace.hermite_next_);
    }

    const double normalization_bra =
        bra.normalized ? primitive_normalization(bra.exponent, bra.angular) : 1.0;
    const double normalization_ket =
        ket.normalized ? primitive_normalization(ket.exponent, ket.angular) : 1.0;
    const double gaussian_integral =
        std::pow(std::numbers::pi / pair.exponent, 1.5);
    return bra.coefficient * ket.coefficient * normalization_bra *
           normalization_ket * pair.prefactor() * gaussian_integral * cartesian;
}

Complex primitive_overlap(const PrimitiveGaussian& bra,
                          const PrimitiveGaussian& ket,
                          const MagneticField& field) {
    IntegralWorkspace workspace;
    return primitive_overlap(bra, ket, field, workspace);
}

std::size_t shell_pair_size(const Shell& a, const Shell& b) noexcept {
    return a.ao_count() * b.ao_count();
}

void compute_overlap(const Shell& a, const Shell& b,
                     const MagneticField& field, std::span<Complex> output,
                     IntegralWorkspace& workspace) {
    if (output.size() != shell_pair_size(a, b)) {
        throw std::length_error("overlap output has an incorrect size");
    }
    std::fill(output.begin(), output.end(), Complex{});

    const auto component_count_a = a.cartesian_count();
    const auto component_count_b = b.cartesian_count();
    workspace.prepare_overlap(
        static_cast<std::size_t>(a.angular_momentum()) + b.angular_momentum(),
        component_count_a * a.primitive_count(),
        component_count_b * b.primitive_count());

    for (std::size_t component = 0; component < component_count_a; ++component) {
        for (std::size_t primitive = 0; primitive < a.primitive_count();
             ++primitive) {
            workspace.normalizations_a_[component * a.primitive_count() + primitive] =
                primitive_normalization(a.exponents()[primitive],
                                        a.components()[component]);
        }
    }
    for (std::size_t component = 0; component < component_count_b; ++component) {
        for (std::size_t primitive = 0; primitive < b.primitive_count();
             ++primitive) {
            workspace.normalizations_b_[component * b.primitive_count() + primitive] =
                primitive_normalization(b.exponents()[primitive],
                                        b.components()[component]);
        }
    }

    for (std::size_t primitive_a = 0; primitive_a < a.primitive_count();
         ++primitive_a) {
        for (std::size_t primitive_b = 0; primitive_b < b.primitive_count();
             ++primitive_b) {
            const PrimitiveGaussian base_a(a.exponents()[primitive_a], a.center(),
                                           {}, 1.0, false);
            const PrimitiveGaussian base_b(b.exponents()[primitive_b], b.center(),
                                           {}, 1.0, false);
            const auto pair = gaussian_product(base_a, base_b, field);
            const double gaussian_integral =
                std::pow(std::numbers::pi / pair.exponent, 1.5);

            for (std::size_t component_a = 0; component_a < component_count_a;
                 ++component_a) {
                for (std::size_t component_b = 0;
                     component_b < component_count_b; ++component_b) {
                    Complex cartesian{1.0, 0.0};
                    for (std::size_t axis = 0; axis < 3; ++axis) {
                        cartesian *= one_dimensional_overlap(
                            a.components()[component_a][axis],
                            b.components()[component_b][axis], a.center()[axis],
                            b.center()[axis], pair.product_center[axis],
                            pair.exponent, pair.pair_wave_vector[axis],
                            workspace.hermite_current_, workspace.hermite_next_);
                    }
                    const Complex primitive_value =
                        pair.prefactor() * gaussian_integral * cartesian *
                        workspace.normalizations_a_[
                            component_a * a.primitive_count() + primitive_a] *
                        workspace.normalizations_b_[
                            component_b * b.primitive_count() + primitive_b];

                    for (std::size_t contraction_a = 0;
                         contraction_a < a.contraction_count(); ++contraction_a) {
                        const std::size_t ao_a =
                            contraction_a * component_count_a + component_a;
                        const double coefficient_a =
                            a.coefficient(contraction_a, primitive_a);
                        for (std::size_t contraction_b = 0;
                             contraction_b < b.contraction_count(); ++contraction_b) {
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

void compute_overlap_matrix(const Basis& basis, const MagneticField& field,
                            std::span<Complex> output) {
    const auto ao_count = basis.ao_count();
    if (ao_count != 0 &&
        ao_count > std::numeric_limits<std::size_t>::max() / ao_count) {
        throw std::length_error("basis overlap size overflow");
    }
    if (output.size() != ao_count * ao_count) {
        throw std::length_error("basis overlap output has an incorrect size");
    }
    std::fill(output.begin(), output.end(), Complex{});
    IntegralWorkspace workspace;
    std::vector<Complex> block;
    for (std::size_t shell_a = 0; shell_a < basis.shells().size(); ++shell_a) {
        const auto& a = basis.shells()[shell_a];
        for (std::size_t shell_b = 0; shell_b < basis.shells().size(); ++shell_b) {
            const auto& b = basis.shells()[shell_b];
            block.resize(shell_pair_size(a, b));
            compute_overlap(a, b, field, block, workspace);
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

}  // namespace giao

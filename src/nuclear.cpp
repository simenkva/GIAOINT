#include "giao_integrals/nuclear.hpp"

#include "hermite.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace giao {
namespace {

std::size_t table_index(std::size_t t, std::size_t u, std::size_t v,
                        std::size_t n, std::size_t side) noexcept {
    return (((n * side + t) * side + u) * side + v);
}

}  // namespace

namespace detail {

struct NuclearKernel {
    static Complex compute(const PrimitiveGaussian& bra,
                           const PrimitiveGaussian& ket,
                           const Nucleus& nucleus,
                           const MagneticField& field,
                           NuclearAttractionWorkspace& workspace);
};

Complex NuclearKernel::compute(const PrimitiveGaussian& bra,
                               const PrimitiveGaussian& ket,
                               const Nucleus& nucleus,
                               const MagneticField& field,
                               NuclearAttractionWorkspace& workspace) {
    const auto pair = gaussian_product(bra, ket, field);
    const std::size_t maximum_order =
        static_cast<std::size_t>(bra.angular.total() + ket.angular.total());
    if (maximum_order > 32U) {
        throw std::invalid_argument(
            "nuclear attraction supports combined angular order through 32");
    }
    workspace.prepare(maximum_order);

    for (std::size_t axis = 0; axis < 3; ++axis) {
        build_hermite_coefficients(
            bra.angular[axis], ket.angular[axis], bra.center[axis],
            ket.center[axis], pair.product_center[axis], pair.exponent,
            workspace.hermite_current_, workspace.hermite_next_);
        const auto axis_order = static_cast<std::size_t>(bra.angular[axis]) +
                                static_cast<std::size_t>(ket.angular[axis]);
        std::copy_n(workspace.hermite_current_.begin(), axis_order + 1U,
                    workspace.hermite_coefficients_[axis].begin());
    }

    const std::array<Complex, 3> displacement{
        pair.complex_center[0] - nucleus.center.x,
        pair.complex_center[1] - nucleus.center.y,
        pair.complex_center[2] - nucleus.center.z};
    const Complex argument =
        pair.exponent * (displacement[0] * displacement[0] +
                         displacement[1] * displacement[1] +
                         displacement[2] * displacement[2]);
    const BoysScaling scaling = argument.real() < 0.0
                                    ? BoysScaling::exp_z
                                    : BoysScaling::unscaled;
    [[maybe_unused]] const auto diagnostics =
        compute_boys(argument,
                     std::span<Complex>(workspace.boys_.data(),
                                        maximum_order + 1U),
                     scaling);

    const double q_squared =
        pair.pair_wave_vector.x * pair.pair_wave_vector.x +
        pair.pair_wave_vector.y * pair.pair_wave_vector.y +
        pair.pair_wave_vector.z * pair.pair_wave_vector.z;
    const double q_dot_p =
        pair.pair_wave_vector.x * pair.product_center.x +
        pair.pair_wave_vector.y * pair.product_center.y +
        pair.pair_wave_vector.z * pair.product_center.z;
    Complex common = pair.london_prefactor;
    if (scaling == BoysScaling::exp_z) {
        common = std::exp(Complex{-q_squared / (4.0 * pair.exponent),
                                  -q_dot_p} -
                          argument);
    }

    const std::size_t side = maximum_order + 1U;
    const std::size_t table_size = side * side * side * side;
    std::fill_n(workspace.coulomb_ready_.begin(), table_size, 0U);
    double radial_factor = 1.0;
    for (std::size_t n = 0; n <= maximum_order; ++n) {
        const auto index = table_index(0, 0, 0, n, side);
        workspace.coulomb_[index] = common * radial_factor * workspace.boys_[n];
        workspace.coulomb_ready_[index] = 1U;
        radial_factor *= -2.0 * pair.exponent;
    }

    const std::array<double, 3> wave{pair.pair_wave_vector.x,
                                     pair.pair_wave_vector.y,
                                     pair.pair_wave_vector.z};
    const auto auxiliary = [&](auto&& self, std::size_t t, std::size_t u,
                               std::size_t v, std::size_t n) -> Complex {
        const auto index = table_index(t, u, v, n, side);
        if (workspace.coulomb_ready_[index] != 0U) {
            return workspace.coulomb_[index];
        }
        Complex value{};
        if (t > 0U) {
            value = Complex{0.0, -wave[0]} * self(self, t - 1U, u, v, n) +
                    displacement[0] * self(self, t - 1U, u, v, n + 1U);
            if (t > 1U) {
                value += static_cast<double>(t - 1U) *
                         self(self, t - 2U, u, v, n + 1U);
            }
        } else if (u > 0U) {
            value = Complex{0.0, -wave[1]} * self(self, t, u - 1U, v, n) +
                    displacement[1] * self(self, t, u - 1U, v, n + 1U);
            if (u > 1U) {
                value += static_cast<double>(u - 1U) *
                         self(self, t, u - 2U, v, n + 1U);
            }
        } else {
            value = Complex{0.0, -wave[2]} * self(self, t, u, v - 1U, n) +
                    displacement[2] * self(self, t, u, v - 1U, n + 1U);
            if (v > 1U) {
                value += static_cast<double>(v - 1U) *
                         self(self, t, u, v - 2U, n + 1U);
            }
        }
        workspace.coulomb_[index] = value;
        workspace.coulomb_ready_[index] = 1U;
        return value;
    };

    Complex contraction{};
    const auto maximum_x = static_cast<std::size_t>(bra.angular.x + ket.angular.x);
    const auto maximum_y = static_cast<std::size_t>(bra.angular.y + ket.angular.y);
    const auto maximum_z = static_cast<std::size_t>(bra.angular.z + ket.angular.z);
    for (std::size_t t = 0; t <= maximum_x; ++t) {
        for (std::size_t u = 0; u <= maximum_y; ++u) {
            for (std::size_t v = 0; v <= maximum_z; ++v) {
                contraction += workspace.hermite_coefficients_[0][t] *
                               workspace.hermite_coefficients_[1][u] *
                               workspace.hermite_coefficients_[2][v] *
                               auxiliary(auxiliary, t, u, v, 0U);
            }
        }
    }

    const double normalization_bra =
        bra.normalized ? primitive_normalization(bra.exponent, bra.angular) : 1.0;
    const double normalization_ket =
        ket.normalized ? primitive_normalization(ket.exponent, ket.angular) : 1.0;
    return -nucleus.charge * (2.0 * std::numbers::pi / pair.exponent) *
           pair.ordinary_prefactor * bra.coefficient * ket.coefficient *
           normalization_bra * normalization_ket * contraction;
}

}  // namespace detail

void NuclearAttractionWorkspace::prepare(
    std::size_t maximum_angular_order, std::size_t normalization_count_a,
    std::size_t normalization_count_b) {
    const std::size_t side = maximum_angular_order + 1U;
    if (side > 0U && side > std::numeric_limits<std::size_t>::max() / side / side /
                              side) {
        throw std::length_error("nuclear-attraction workspace size overflow");
    }
    hermite_current_.resize(
        std::max(hermite_current_.size(), maximum_angular_order + 2U));
    hermite_next_.resize(
        std::max(hermite_next_.size(), maximum_angular_order + 2U));
    for (auto& coefficients : hermite_coefficients_) {
        coefficients.resize(std::max(coefficients.size(), side));
    }
    normalizations_a_.resize(
        std::max(normalizations_a_.size(), normalization_count_a));
    normalizations_b_.resize(
        std::max(normalizations_b_.size(), normalization_count_b));
    boys_.resize(std::max(boys_.size(), side));
    const auto table_size = side * side * side * side;
    coulomb_.resize(std::max(coulomb_.size(), table_size));
    coulomb_ready_.resize(std::max(coulomb_ready_.size(), table_size));
}

Complex primitive_nuclear_attraction(
    const PrimitiveGaussian& bra, const PrimitiveGaussian& ket,
    std::span<const Nucleus> nuclei, const MagneticField& field,
    NuclearAttractionWorkspace& workspace) {
    Complex result{};
    for (const auto& nucleus : nuclei) {
        result += detail::NuclearKernel::compute(bra, ket, nucleus, field,
                                                 workspace);
    }
    return result;
}

Complex primitive_nuclear_attraction(const PrimitiveGaussian& bra,
                                     const PrimitiveGaussian& ket,
                                     std::span<const Nucleus> nuclei,
                                     const MagneticField& field) {
    NuclearAttractionWorkspace workspace;
    return primitive_nuclear_attraction(bra, ket, nuclei, field, workspace);
}

void compute_nuclear_attraction(
    const Shell& a, const Shell& b, std::span<const Nucleus> nuclei,
    const MagneticField& field, std::span<Complex> output,
    NuclearAttractionWorkspace& workspace) {
    if (output.size() != shell_pair_size(a, b)) {
        throw std::length_error(
            "nuclear-attraction output has an incorrect size");
    }
    std::fill(output.begin(), output.end(), Complex{});
    const auto component_count_a = a.cartesian_count();
    const auto component_count_b = b.cartesian_count();
    workspace.prepare(
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
            for (std::size_t component_a = 0; component_a < component_count_a;
                 ++component_a) {
                for (std::size_t component_b = 0;
                     component_b < component_count_b; ++component_b) {
                    const PrimitiveGaussian bra(
                        a.exponents()[primitive_a], a.center(),
                        a.components()[component_a], 1.0, false);
                    const PrimitiveGaussian ket(
                        b.exponents()[primitive_b], b.center(),
                        b.components()[component_b], 1.0, false);
                    const Complex primitive_value =
                        primitive_nuclear_attraction(bra, ket, nuclei, field,
                                                     workspace) *
                        workspace.normalizations_a_[
                            component_a * a.primitive_count() + primitive_a] *
                        workspace.normalizations_b_[
                            component_b * b.primitive_count() + primitive_b];
                    for (std::size_t contraction_a = 0;
                         contraction_a < a.contraction_count(); ++contraction_a) {
                        const std::size_t ao_a =
                            contraction_a * component_count_a + component_a;
                        for (std::size_t contraction_b = 0;
                             contraction_b < b.contraction_count();
                             ++contraction_b) {
                            const std::size_t ao_b =
                                contraction_b * component_count_b + component_b;
                            output[ao_a * b.ao_count() + ao_b] +=
                                a.coefficient(contraction_a, primitive_a) *
                                b.coefficient(contraction_b, primitive_b) *
                                primitive_value;
                        }
                    }
                }
            }
        }
    }
}

void compute_nuclear_attraction_matrix(
    const Basis& basis, std::span<const Nucleus> nuclei,
    const MagneticField& field, std::span<Complex> output) {
    const auto ao_count = basis.ao_count();
    if (ao_count != 0U &&
        ao_count > std::numeric_limits<std::size_t>::max() / ao_count) {
        throw std::length_error("basis nuclear-attraction size overflow");
    }
    if (output.size() != ao_count * ao_count) {
        throw std::length_error(
            "basis nuclear-attraction output has an incorrect size");
    }
    std::fill(output.begin(), output.end(), Complex{});
    NuclearAttractionWorkspace workspace;
    std::vector<Complex> block;
    for (std::size_t shell_a = 0; shell_a < basis.shells().size(); ++shell_a) {
        const auto& a = basis.shells()[shell_a];
        for (std::size_t shell_b = 0; shell_b < basis.shells().size(); ++shell_b) {
            const auto& b = basis.shells()[shell_b];
            block.resize(shell_pair_size(a, b));
            compute_nuclear_attraction(a, b, nuclei, field, block, workspace);
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

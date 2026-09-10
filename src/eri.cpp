#include "giao_integrals/eri.hpp"

#include "hermite.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <complex>
#include <exception>
#include <limits>
#include <mutex>
#include <numbers>
#include <stdexcept>

#ifdef GIAO_HAS_OPENMP
#include <omp.h>
#endif

namespace giao {
namespace {

constexpr std::size_t maximum_auxiliary_entries = 4'000'000U;

std::size_t checked_product(std::span<const std::size_t> factors,
                            const char* message) {
    std::size_t result = 1U;
    for (const auto factor : factors) {
        if (factor != 0U &&
            result > std::numeric_limits<std::size_t>::max() / factor) {
            throw std::length_error(message);
        }
        result *= factor;
    }
    return result;
}

std::size_t auxiliary_index(const std::array<std::size_t, 7>& indices,
                            const std::array<std::size_t, 7>& dimensions) {
    std::size_t result = indices[0];
    for (std::size_t axis = 1; axis < dimensions.size(); ++axis) {
        result = result * dimensions[axis] + indices[axis];
    }
    return result;
}

std::size_t quartet_flat_index(std::size_t a, std::size_t b, std::size_t c,
                               std::size_t d,
                               const std::array<std::size_t, 4>& shape) {
    return ((a * shape[1] + b) * shape[2] + c) * shape[3] + d;
}

double dot(Vec3 left, Vec3 right) noexcept {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

struct OwnedShellQuartetBlock {
    ShellQuartetIndex shells{};
    std::array<std::size_t, 4> shape{};
    std::vector<Complex> values;
};

void validate_quartet(const Basis& basis, ShellQuartetIndex quartet) {
    for (const auto index : quartet.as_array()) {
        if (index >= basis.shells().size()) {
            throw std::out_of_range("ERI shell-quartet index is out of range");
        }
    }
}

}  // namespace

namespace detail {

struct EriKernel {
    static Complex compute(const PrimitiveGaussian& a,
                           const PrimitiveGaussian& b,
                           const PrimitiveGaussian& c,
                           const PrimitiveGaussian& d,
                           const MagneticField& field,
                           EriWorkspace& workspace);
};

Complex EriKernel::compute(const PrimitiveGaussian& a,
                           const PrimitiveGaussian& b,
                           const PrimitiveGaussian& c,
                           const PrimitiveGaussian& d,
                           const MagneticField& field,
                           EriWorkspace& workspace) {
    const auto pair_ab = gaussian_product(a, b, field);
    const auto pair_cd = gaussian_product(c, d, field);
    const std::array<std::size_t, 6> maxima{
        static_cast<std::size_t>(a.angular.x + b.angular.x),
        static_cast<std::size_t>(a.angular.y + b.angular.y),
        static_cast<std::size_t>(a.angular.z + b.angular.z),
        static_cast<std::size_t>(c.angular.x + d.angular.x),
        static_cast<std::size_t>(c.angular.y + d.angular.y),
        static_cast<std::size_t>(c.angular.z + d.angular.z)};
    const std::size_t maximum_order =
        static_cast<std::size_t>(a.angular.total() + b.angular.total() +
                                 c.angular.total() + d.angular.total());
    if (maximum_order > 32U) {
        throw std::invalid_argument(
            "electron repulsion supports combined angular order through 32");
    }
    const std::array<std::size_t, 7> dimensions{
        maximum_order + 1U, maxima[0] + 1U, maxima[1] + 1U,
        maxima[2] + 1U, maxima[3] + 1U, maxima[4] + 1U,
        maxima[5] + 1U};
    workspace.prepare(dimensions, maximum_order);

    const std::array<const PrimitiveGaussian*, 4> primitives{&a, &b, &c, &d};
    const std::array<const GaussianPairData*, 2> pairs{&pair_ab, &pair_cd};
    for (std::size_t pair_index = 0; pair_index < 2; ++pair_index) {
        const auto& left = *primitives[2U * pair_index];
        const auto& right = *primitives[2U * pair_index + 1U];
        const auto& pair = *pairs[pair_index];
        const std::array<std::uint16_t, 6> angular_key{
            left.angular.x, left.angular.y, left.angular.z,
            right.angular.x, right.angular.y, right.angular.z};
        const std::array<double, 10> geometry_key{
            left.center.x,
            left.center.y,
            left.center.z,
            right.center.x,
            right.center.y,
            right.center.z,
            pair.product_center.x,
            pair.product_center.y,
            pair.product_center.z,
            pair.exponent};
        if (workspace.hermite_cache_valid_[pair_index] &&
            workspace.hermite_angular_key_[pair_index] == angular_key &&
            workspace.hermite_geometry_key_[pair_index] == geometry_key) {
            continue;
        }
        for (std::size_t axis = 0; axis < 3; ++axis) {
            build_hermite_coefficients(
                left.angular[axis], right.angular[axis], left.center[axis],
                right.center[axis], pair.product_center[axis], pair.exponent,
                workspace.hermite_current_, workspace.hermite_next_);
            const std::size_t coefficient_index = 3U * pair_index + axis;
            const std::size_t order =
                static_cast<std::size_t>(left.angular[axis]) +
                static_cast<std::size_t>(right.angular[axis]);
            std::copy_n(workspace.hermite_current_.begin(), order + 1U,
                        workspace.hermite_coefficients_[coefficient_index].begin());
        }
        workspace.hermite_angular_key_[pair_index] = angular_key;
        workspace.hermite_geometry_key_[pair_index] = geometry_key;
        workspace.hermite_cache_valid_[pair_index] = true;
        const auto x_extent = maxima[3U * pair_index] + 1U;
        const auto y_extent = maxima[3U * pair_index + 1U] + 1U;
        const auto z_extent = maxima[3U * pair_index + 2U] + 1U;
        auto& products = workspace.pair_hermite_products_[pair_index];
        products.resize(x_extent * y_extent * z_extent);
        for (std::size_t x = 0; x < x_extent; ++x) {
            for (std::size_t y = 0; y < y_extent; ++y) {
                for (std::size_t z = 0; z < z_extent; ++z) {
                    products[(x * y_extent + y) * z_extent + z] =
                        workspace.hermite_coefficients_[3U * pair_index][x] *
                        workspace.hermite_coefficients_[3U * pair_index + 1U][y] *
                        workspace.hermite_coefficients_[3U * pair_index + 2U][z];
                }
            }
        }
    }

    const double p = pair_ab.exponent;
    const double q = pair_cd.exponent;
    const double rho = p * q / (p + q);
    const std::array<Complex, 3> displacement{
        pair_ab.complex_center[0] - pair_cd.complex_center[0],
        pair_ab.complex_center[1] - pair_cd.complex_center[1],
        pair_ab.complex_center[2] - pair_cd.complex_center[2]};
    const Complex argument =
        rho * (displacement[0] * displacement[0] +
               displacement[1] * displacement[1] +
               displacement[2] * displacement[2]);
    const BoysScaling scaling = argument.real() < 0.0
                                    ? BoysScaling::exp_z
                                    : BoysScaling::unscaled;
    if (!workspace.boys_cache_valid_ || workspace.boys_argument_ != argument ||
        workspace.boys_scaling_ != scaling ||
        workspace.boys_maximum_order_ < maximum_order) {
        [[maybe_unused]] const auto diagnostics = compute_boys(
            argument,
            std::span<Complex>(workspace.boys_.data(), maximum_order + 1U),
            scaling);
        workspace.boys_argument_ = argument;
        workspace.boys_scaling_ = scaling;
        workspace.boys_maximum_order_ = maximum_order;
        workspace.boys_cache_valid_ = true;
    }

    Complex common = pair_ab.london_prefactor * pair_cd.london_prefactor;
    if (scaling == BoysScaling::exp_z) {
        const Vec3 real_displacement{
            pair_ab.product_center.x - pair_cd.product_center.x,
            pair_ab.product_center.y - pair_cd.product_center.y,
            pair_ab.product_center.z - pair_cd.product_center.z};
        const Vec3 total_wave{
            pair_ab.pair_wave_vector.x + pair_cd.pair_wave_vector.x,
            pair_ab.pair_wave_vector.y + pair_cd.pair_wave_vector.y,
            pair_ab.pair_wave_vector.z + pair_cd.pair_wave_vector.z};
        const Vec3 weighted_center{
            (p * pair_ab.product_center.x + q * pair_cd.product_center.x) /
                (p + q),
            (p * pair_ab.product_center.y + q * pair_cd.product_center.y) /
                (p + q),
            (p * pair_ab.product_center.z + q * pair_cd.product_center.z) /
                (p + q)};
        const double real_part =
            -rho * dot(real_displacement, real_displacement) -
            dot(total_wave, total_wave) / (4.0 * (p + q));
        common = std::exp(Complex{real_part, -dot(total_wave, weighted_center)});
    }

    ++workspace.auxiliary_generation_;
    if (workspace.auxiliary_generation_ == 0U) {
        std::fill(workspace.auxiliary_generation_tags_.begin(),
                  workspace.auxiliary_generation_tags_.end(), 0U);
        ++workspace.auxiliary_generation_;
    }
    const auto generation = workspace.auxiliary_generation_;
    double radial_factor = 1.0;
    for (std::size_t n = 0; n <= maximum_order; ++n) {
        const std::array<std::size_t, 7> indices{n, 0U, 0U, 0U,
                                                0U, 0U, 0U};
        const auto index = auxiliary_index(indices, dimensions);
        workspace.auxiliary_[index] = common * radial_factor * workspace.boys_[n];
        workspace.auxiliary_generation_tags_[index] = generation;
        radial_factor *= -2.0 * rho;
    }

    const std::array<double, 3> wave_ab{pair_ab.pair_wave_vector.x,
                                        pair_ab.pair_wave_vector.y,
                                        pair_ab.pair_wave_vector.z};
    const std::array<double, 3> wave_cd{pair_cd.pair_wave_vector.x,
                                        pair_cd.pair_wave_vector.y,
                                        pair_cd.pair_wave_vector.z};
    const auto auxiliary =
        [&](auto&& self, std::size_t n, std::size_t t, std::size_t u,
            std::size_t v, std::size_t tau, std::size_t phi,
            std::size_t chi) -> Complex {
        const std::array<std::size_t, 7> indices{n, t, u, v, tau, phi, chi};
        const auto index = auxiliary_index(indices, dimensions);
        if (workspace.auxiliary_generation_tags_[index] == generation) {
            return workspace.auxiliary_[index];
        }
        const std::array<std::size_t, 3> first{t, u, v};
        const std::array<std::size_t, 3> second{tau, phi, chi};
        Complex value{};
        bool evaluated = false;
        for (std::size_t axis = 0; axis < 3 && !evaluated; ++axis) {
            if (first[axis] == 0U) {
                continue;
            }
            auto lower_first = first;
            --lower_first[axis];
            value = Complex{0.0, -wave_ab[axis]} *
                    self(self, n, lower_first[0], lower_first[1],
                         lower_first[2], second[0], second[1], second[2]);
            value += displacement[axis] *
                     self(self, n + 1U, lower_first[0], lower_first[1],
                          lower_first[2], second[0], second[1], second[2]);
            if (lower_first[axis] > 0U) {
                auto twice_lower = lower_first;
                --twice_lower[axis];
                value += static_cast<double>(lower_first[axis]) *
                         self(self, n + 1U, twice_lower[0], twice_lower[1],
                              twice_lower[2], second[0], second[1], second[2]);
            }
            if (second[axis] > 0U) {
                auto lower_second = second;
                --lower_second[axis];
                value -= static_cast<double>(second[axis]) *
                         self(self, n + 1U, lower_first[0], lower_first[1],
                              lower_first[2], lower_second[0], lower_second[1],
                              lower_second[2]);
            }
            evaluated = true;
        }
        for (std::size_t axis = 0; axis < 3 && !evaluated; ++axis) {
            if (second[axis] == 0U) {
                continue;
            }
            auto lower_second = second;
            --lower_second[axis];
            value = Complex{0.0, -wave_cd[axis]} *
                    self(self, n, first[0], first[1], first[2],
                         lower_second[0], lower_second[1], lower_second[2]);
            value -= displacement[axis] *
                     self(self, n + 1U, first[0], first[1], first[2],
                          lower_second[0], lower_second[1], lower_second[2]);
            if (first[axis] > 0U) {
                auto lower_first = first;
                --lower_first[axis];
                value -= static_cast<double>(first[axis]) *
                         self(self, n + 1U, lower_first[0], lower_first[1],
                              lower_first[2], lower_second[0], lower_second[1],
                              lower_second[2]);
            }
            if (lower_second[axis] > 0U) {
                auto twice_lower = lower_second;
                --twice_lower[axis];
                value += static_cast<double>(lower_second[axis]) *
                         self(self, n + 1U, first[0], first[1], first[2],
                              twice_lower[0], twice_lower[1], twice_lower[2]);
            }
            evaluated = true;
        }
        if (!evaluated) {
            throw std::logic_error("invalid ERI auxiliary recurrence state");
        }
        workspace.auxiliary_[index] = value;
        workspace.auxiliary_generation_tags_[index] = generation;
        return value;
    };

    Complex contraction{};
    const auto ab_y_extent = maxima[1] + 1U;
    const auto ab_z_extent = maxima[2] + 1U;
    const auto cd_y_extent = maxima[4] + 1U;
    const auto cd_z_extent = maxima[5] + 1U;
    for (std::size_t t = 0; t <= maxima[0]; ++t) {
        for (std::size_t u = 0; u <= maxima[1]; ++u) {
            for (std::size_t v = 0; v <= maxima[2]; ++v) {
                const double coefficient_ab =
                    workspace.pair_hermite_products_[0]
                        [(t * ab_y_extent + u) * ab_z_extent + v];
                for (std::size_t tau = 0; tau <= maxima[3]; ++tau) {
                    for (std::size_t phi = 0; phi <= maxima[4]; ++phi) {
                        for (std::size_t chi = 0; chi <= maxima[5]; ++chi) {
                            contraction +=
                                coefficient_ab *
                                workspace.pair_hermite_products_[1]
                                    [(tau * cd_y_extent + phi) * cd_z_extent +
                                     chi] *
                                auxiliary(auxiliary, 0U, t, u, v, tau, phi,
                                          chi);
                        }
                    }
                }
            }
        }
    }

    const auto normalization = [](const PrimitiveGaussian& primitive) {
        return primitive.normalized
                   ? primitive_normalization(primitive.exponent,
                                             primitive.angular)
                   : 1.0;
    };
    const double prefactor =
        2.0 * std::pow(std::numbers::pi, 2.5) /
        (p * q * std::sqrt(p + q));
    return prefactor * pair_ab.ordinary_prefactor * pair_cd.ordinary_prefactor *
           a.coefficient * b.coefficient * c.coefficient * d.coefficient *
           normalization(a) * normalization(b) * normalization(c) *
           normalization(d) * contraction;
}

}  // namespace detail

void EriWorkspace::prepare(
    std::array<std::size_t, 7> auxiliary_dimensions,
    std::size_t maximum_hermite_order,
    std::array<std::size_t, 4> normalization_counts) {
    const auto table_size =
        checked_product(auxiliary_dimensions, "ERI auxiliary size overflow");
    if (table_size > maximum_auxiliary_entries) {
        throw std::length_error(
            "ERI auxiliary exceeds the Milestone 5 workspace limit");
    }
    hermite_current_.resize(
        std::max(hermite_current_.size(), maximum_hermite_order + 2U));
    hermite_next_.resize(
        std::max(hermite_next_.size(), maximum_hermite_order + 2U));
    for (auto& coefficients : hermite_coefficients_) {
        coefficients.resize(
            std::max(coefficients.size(), maximum_hermite_order + 1U));
    }
    for (std::size_t center = 0; center < 4; ++center) {
        normalizations_[center].resize(std::max(normalizations_[center].size(),
                                                normalization_counts[center]));
    }
    boys_.resize(std::max(boys_.size(), auxiliary_dimensions[0]));
    auxiliary_.resize(std::max(auxiliary_.size(), table_size));
    auxiliary_generation_tags_.resize(
        std::max(auxiliary_generation_tags_.size(), table_size));
}

CanonicalShellQuartet canonicalize_shell_quartet(
    ShellQuartetIndex quartet) noexcept {
    const std::array<std::array<std::uint32_t, 4>, 4> candidates{
        quartet.as_array(),
        std::array<std::uint32_t, 4>{quartet.c, quartet.d, quartet.a, quartet.b},
        std::array<std::uint32_t, 4>{quartet.b, quartet.a, quartet.d, quartet.c},
        std::array<std::uint32_t, 4>{quartet.d, quartet.c, quartet.b, quartet.a}};
    std::size_t selected = 0U;
    for (std::size_t index = 1; index < candidates.size(); ++index) {
        if (candidates[index] < candidates[selected]) {
            selected = index;
        }
    }
    const auto& value = candidates[selected];
    return {{value[0], value[1], value[2], value[3]}, selected >= 2U};
}

std::size_t shell_quartet_size(const Shell& a, const Shell& b,
                               const Shell& c, const Shell& d) {
    const std::array<std::size_t, 4> dimensions{a.ao_count(), b.ao_count(),
                                                c.ao_count(), d.ao_count()};
    return checked_product(dimensions, "shell quartet size overflow");
}

Complex primitive_eri(const PrimitiveGaussian& a, const PrimitiveGaussian& b,
                      const PrimitiveGaussian& c, const PrimitiveGaussian& d,
                      const MagneticField& field, EriWorkspace& workspace) {
    return detail::EriKernel::compute(a, b, c, d, field, workspace);
}

Complex primitive_eri(const PrimitiveGaussian& a, const PrimitiveGaussian& b,
                      const PrimitiveGaussian& c, const PrimitiveGaussian& d,
                      const MagneticField& field) {
    EriWorkspace workspace;
    return primitive_eri(a, b, c, d, field, workspace);
}

void compute_eri(const Shell& a, const Shell& b, const Shell& c,
                 const Shell& d, const MagneticField& field,
                 std::span<Complex> output, EriWorkspace& workspace) {
    const std::array<const Shell*, 4> shells{&a, &b, &c, &d};
    const std::array<std::size_t, 4> shape{a.ao_count(), b.ao_count(),
                                           c.ao_count(), d.ao_count()};
    if (output.size() != shell_quartet_size(a, b, c, d)) {
        throw std::length_error("ERI output has an incorrect size");
    }
    std::fill(output.begin(), output.end(), Complex{});
    const std::size_t maximum_order =
        static_cast<std::size_t>(a.angular_momentum()) + b.angular_momentum() +
        c.angular_momentum() + d.angular_momentum();
    const std::array<std::size_t, 4> normalization_counts{
        a.cartesian_count() * a.primitive_count(),
        b.cartesian_count() * b.primitive_count(),
        c.cartesian_count() * c.primitive_count(),
        d.cartesian_count() * d.primitive_count()};
    workspace.prepare({1U, 1U, 1U, 1U, 1U, 1U, 1U}, maximum_order,
                      normalization_counts);
    for (std::size_t center = 0; center < 4; ++center) {
        const auto& shell = *shells[center];
        for (std::size_t component = 0; component < shell.cartesian_count();
             ++component) {
            for (std::size_t primitive = 0; primitive < shell.primitive_count();
                 ++primitive) {
                workspace.normalizations_[center]
                                         [component * shell.primitive_count() +
                                          primitive] =
                    primitive_normalization(shell.exponents()[primitive],
                                            shell.components()[component]);
            }
        }
    }

    for (std::size_t pa = 0; pa < a.primitive_count(); ++pa) {
        for (std::size_t pb = 0; pb < b.primitive_count(); ++pb) {
            for (std::size_t pc = 0; pc < c.primitive_count(); ++pc) {
                for (std::size_t pd = 0; pd < d.primitive_count(); ++pd) {
                    const std::array<std::size_t, 4> primitive_indices{pa, pb,
                                                                      pc, pd};
                    for (std::size_t ca = 0; ca < a.cartesian_count(); ++ca) {
                        for (std::size_t cb = 0; cb < b.cartesian_count(); ++cb) {
                            for (std::size_t cc = 0; cc < c.cartesian_count();
                                 ++cc) {
                                for (std::size_t cd = 0;
                                     cd < d.cartesian_count(); ++cd) {
                                    const std::array<std::size_t, 4> components{
                                        ca, cb, cc, cd};
                                    const std::array<PrimitiveGaussian, 4>
                                        primitives{
                                            PrimitiveGaussian(a.exponents()[pa],
                                                              a.center(),
                                                              a.components()[ca],
                                                              1.0, false),
                                            PrimitiveGaussian(b.exponents()[pb],
                                                              b.center(),
                                                              b.components()[cb],
                                                              1.0, false),
                                            PrimitiveGaussian(c.exponents()[pc],
                                                              c.center(),
                                                              c.components()[cc],
                                                              1.0, false),
                                            PrimitiveGaussian(d.exponents()[pd],
                                                              d.center(),
                                                              d.components()[cd],
                                                              1.0, false)};
                                    Complex primitive_value = primitive_eri(
                                        primitives[0], primitives[1],
                                        primitives[2], primitives[3], field,
                                        workspace);
                                    for (std::size_t center = 0; center < 4;
                                         ++center) {
                                        const auto& shell = *shells[center];
                                        primitive_value *=
                                            workspace.normalizations_[center]
                                                [components[center] *
                                                     shell.primitive_count() +
                                                 primitive_indices[center]];
                                    }
                                    for (std::size_t xa = 0;
                                         xa < a.contraction_count(); ++xa) {
                                        const auto ao_a =
                                            xa * a.cartesian_count() + ca;
                                        for (std::size_t xb = 0;
                                             xb < b.contraction_count(); ++xb) {
                                            const auto ao_b =
                                                xb * b.cartesian_count() + cb;
                                            for (std::size_t xc = 0;
                                                 xc < c.contraction_count();
                                                 ++xc) {
                                                const auto ao_c =
                                                    xc * c.cartesian_count() + cc;
                                                for (std::size_t xd = 0;
                                                     xd < d.contraction_count();
                                                     ++xd) {
                                                    const auto ao_d =
                                                        xd * d.cartesian_count() +
                                                        cd;
                                                    output[quartet_flat_index(
                                                        ao_a, ao_b, ao_c, ao_d,
                                                        shape)] +=
                                                        a.coefficient(xa, pa) *
                                                        b.coefficient(xb, pb) *
                                                        c.coefficient(xc, pc) *
                                                        d.coefficient(xd, pd) *
                                                        primitive_value;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

EriSchwarzBounds::EriSchwarzBounds(const Basis& basis,
                                   const MagneticField& field)
    : basis_identity_(&basis), field_(field), shell_count_(basis.shells().size()),
      values_(shell_count_ * shell_count_, 0.0) {
    EriWorkspace workspace;
    std::vector<Complex> block;
    constexpr double safety =
        1.0 + 64.0 * std::numeric_limits<double>::epsilon();
    for (std::size_t a_index = 0; a_index < shell_count_; ++a_index) {
        for (std::size_t b_index = a_index; b_index < shell_count_; ++b_index) {
            const auto& a = basis.shells()[a_index];
            const auto& b = basis.shells()[b_index];
            const std::array<std::size_t, 4> shape{
                a.ao_count(), b.ao_count(), b.ao_count(), a.ao_count()};
            block.resize(shell_quartet_size(a, b, b, a));
            compute_eri(a, b, b, a, field, block, workspace);
            double maximum_norm = 0.0;
            for (std::size_t ia = 0; ia < a.ao_count(); ++ia) {
                for (std::size_t ib = 0; ib < b.ao_count(); ++ib) {
                    maximum_norm = std::max(
                        maximum_norm,
                        std::abs(block[quartet_flat_index(ia, ib, ib, ia,
                                                          shape)]));
                }
            }
            const double bound =
                std::nextafter(std::sqrt(maximum_norm) * safety,
                               std::numeric_limits<double>::infinity());
            values_[a_index * shell_count_ + b_index] = bound;
            values_[b_index * shell_count_ + a_index] = bound;
        }
    }
}

double EriSchwarzBounds::operator()(std::size_t a, std::size_t b) const {
    if (a >= shell_count_ || b >= shell_count_) {
        throw std::out_of_range("ERI Schwarz shell-pair index is out of range");
    }
    return values_[a * shell_count_ + b];
}

bool EriSchwarzBounds::matches(const Basis& basis,
                               const MagneticField& field) const noexcept {
    return basis_identity_ == &basis && field_.B.x == field.B.x &&
           field_.B.y == field.B.y && field_.B.z == field.B.z &&
           field_.gauge_origin.x == field.gauge_origin.x &&
           field_.gauge_origin.y == field.gauge_origin.y &&
           field_.gauge_origin.z == field.gauge_origin.z;
}

void for_each_eri_shell_quartet(
    const Basis& basis, std::span<const ShellQuartetIndex> quartets,
    const MagneticField& field, QuartetConsumer consumer, void* user_data) {
    [[maybe_unused]] const auto statistics = evaluate_eri_shell_quartets(
        basis, quartets, field, {}, nullptr, consumer, user_data);
}

EriStatistics evaluate_eri_shell_quartets(
    const Basis& basis, std::span<const ShellQuartetIndex> quartets,
    const MagneticField& field, const EriEvaluationOptions& options,
    const EriSchwarzBounds* bounds, QuartetConsumer consumer,
    void* user_data) {
    if (consumer == nullptr) {
        throw std::invalid_argument("ERI quartet consumer must not be null");
    }
    if (!std::isfinite(options.screening_threshold) ||
        options.screening_threshold < 0.0) {
        throw std::invalid_argument(
            "ERI screening threshold must be finite and non-negative");
    }
    if (options.thread_count == 0U) {
        throw std::invalid_argument("ERI thread count must be positive");
    }
    if (options.screening_threshold > 0.0 && bounds == nullptr) {
        throw std::invalid_argument(
            "positive ERI screening requires Schwarz bounds");
    }
    if (bounds != nullptr && !bounds->matches(basis, field)) {
        throw std::invalid_argument("ERI Schwarz bounds do not match the basis");
    }
#ifndef GIAO_HAS_OPENMP
    if (options.thread_count != 1U) {
        throw std::invalid_argument(
            "parallel ERI evaluation requires an OpenMP-enabled build");
    }
#endif

    EriStatistics statistics{quartets.size(), 0U, 0U};
    std::vector<ShellQuartetIndex> active;
    active.reserve(quartets.size());
    for (const auto quartet : quartets) {
        validate_quartet(basis, quartet);
        if (options.screening_threshold > 0.0 &&
            (*bounds)(quartet.a, quartet.b) *
                    (*bounds)(quartet.c, quartet.d) <
                options.screening_threshold) {
            ++statistics.screened_quartets;
            continue;
        }
        active.push_back(quartet);
    }
    statistics.computed_quartets = active.size();

    if (options.thread_count == 1U || active.size() < 2U) {
        EriWorkspace workspace;
        std::vector<Complex> block;
        for (const auto quartet : active) {
            const auto& a = basis.shells()[quartet.a];
            const auto& b = basis.shells()[quartet.b];
            const auto& c = basis.shells()[quartet.c];
            const auto& d = basis.shells()[quartet.d];
            block.resize(shell_quartet_size(a, b, c, d));
            compute_eri(a, b, c, d, field, block, workspace);
            consumer({quartet,
                      {a.ao_count(), b.ao_count(), c.ao_count(), d.ao_count()},
                      block},
                     user_data);
        }
        return statistics;
    }

#ifdef GIAO_HAS_OPENMP
    if (options.thread_count >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("ERI thread count is too large");
    }
    std::vector<OwnedShellQuartetBlock> blocks(active.size());
    for (std::size_t index = 0; index < active.size(); ++index) {
        const auto quartet = active[index];
        const auto& a = basis.shells()[quartet.a];
        const auto& b = basis.shells()[quartet.b];
        const auto& c = basis.shells()[quartet.c];
        const auto& d = basis.shells()[quartet.d];
        blocks[index].shells = quartet;
        blocks[index].shape = {a.ao_count(), b.ao_count(), c.ao_count(),
                               d.ao_count()};
        blocks[index].values.resize(shell_quartet_size(a, b, c, d));
    }

    std::exception_ptr failure;
    std::mutex failure_mutex;
    std::atomic<bool> failed{false};
    const int requested_threads = static_cast<int>(options.thread_count);
#pragma omp parallel num_threads(requested_threads)
    {
        EriWorkspace workspace;
#pragma omp for schedule(static)
        for (std::ptrdiff_t signed_index = 0;
             signed_index < static_cast<std::ptrdiff_t>(blocks.size());
             ++signed_index) {
            if (failed.load(std::memory_order_relaxed)) {
                continue;
            }
            const auto index = static_cast<std::size_t>(signed_index);
            const auto quartet = blocks[index].shells;
            try {
                compute_eri(basis.shells()[quartet.a],
                            basis.shells()[quartet.b],
                            basis.shells()[quartet.c],
                            basis.shells()[quartet.d], field,
                            blocks[index].values, workspace);
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
                std::scoped_lock lock(failure_mutex);
                if (failure == nullptr) {
                    failure = std::current_exception();
                }
            }
        }
    }
    if (failure != nullptr) {
        std::rethrow_exception(failure);
    }
    for (const auto& block : blocks) {
        consumer({block.shells, block.shape, block.values}, user_data);
    }
    return statistics;
#else
    return statistics;
#endif
}

namespace {

struct TensorConsumerData {
    const Basis* basis{};
    std::span<Complex> output{};
    std::array<std::size_t, 4> shape{};
};

void write_tensor_block(const ShellQuartetBlockView& block, void* user_data) {
    auto& data = *static_cast<TensorConsumerData*>(user_data);
    const auto& basis = *data.basis;
    const auto [sa, sb, sc, sd] = block.shells.as_array();
    for (std::size_t ia = 0; ia < block.shape[0]; ++ia) {
        for (std::size_t ib = 0; ib < block.shape[1]; ++ib) {
            for (std::size_t ic = 0; ic < block.shape[2]; ++ic) {
                for (std::size_t id = 0; id < block.shape[3]; ++id) {
                    const auto value = block.values[quartet_flat_index(
                        ia, ib, ic, id, block.shape)];
                    const auto ga = basis.ao_offsets()[sa] + ia;
                    const auto gb = basis.ao_offsets()[sb] + ib;
                    const auto gc = basis.ao_offsets()[sc] + ic;
                    const auto gd = basis.ao_offsets()[sd] + id;
                    data.output[quartet_flat_index(ga, gb, gc, gd,
                                                   data.shape)] = value;
                    data.output[quartet_flat_index(gc, gd, ga, gb,
                                                   data.shape)] = value;
                    data.output[quartet_flat_index(gb, ga, gd, gc,
                                                   data.shape)] =
                        std::conj(value);
                    data.output[quartet_flat_index(gd, gc, gb, ga,
                                                   data.shape)] =
                        std::conj(value);
                }
            }
        }
    }
}

}  // namespace

EriStatistics compute_eri_tensor(const Basis& basis,
                                 const MagneticField& field,
                                 const EriEvaluationOptions& options,
                                 const EriSchwarzBounds* bounds,
                                 std::span<Complex> output) {
    const std::array<std::size_t, 4> tensor_shape{
        basis.ao_count(), basis.ao_count(), basis.ao_count(), basis.ao_count()};
    const auto required = checked_product(tensor_shape, "basis ERI size overflow");
    if (output.size() != required) {
        throw std::length_error("basis ERI output has an incorrect size");
    }
    std::fill(output.begin(), output.end(), Complex{});
    const auto shell_count = basis.shells().size();
    if (shell_count > std::numeric_limits<std::uint32_t>::max()) {
        throw std::length_error("basis has too many shells for ERI indices");
    }
    std::vector<ShellQuartetIndex> quartets;
    for (std::size_t sa = 0; sa < shell_count; ++sa) {
        for (std::size_t sb = 0; sb < shell_count; ++sb) {
            for (std::size_t sc = 0; sc < shell_count; ++sc) {
                for (std::size_t sd = 0; sd < shell_count; ++sd) {
                    const ShellQuartetIndex quartet{
                        static_cast<std::uint32_t>(sa),
                        static_cast<std::uint32_t>(sb),
                        static_cast<std::uint32_t>(sc),
                        static_cast<std::uint32_t>(sd)};
                    if (canonicalize_shell_quartet(quartet).shells.as_array() !=
                        quartet.as_array()) {
                        continue;
                    }
                    quartets.push_back(quartet);
                }
            }
        }
    }
    TensorConsumerData data{&basis, output, tensor_shape};
    return evaluate_eri_shell_quartets(basis, quartets, field, options, bounds,
                                       write_tensor_block, &data);
}

void compute_eri_tensor(const Basis& basis, const MagneticField& field,
                        std::span<Complex> output) {
    [[maybe_unused]] const auto statistics =
        compute_eri_tensor(basis, field, {}, nullptr, output);
}

bool openmp_enabled() noexcept {
#ifdef GIAO_HAS_OPENMP
    return true;
#else
    return false;
#endif
}

std::size_t openmp_max_threads() noexcept {
#ifdef GIAO_HAS_OPENMP
    return static_cast<std::size_t>(omp_get_max_threads());
#else
    return 1U;
#endif
}

}  // namespace giao

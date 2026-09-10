#include "giao_integrals/boys.hpp"
#include "giao_integrals/derivatives.hpp"
#include "giao_integrals/eri.hpp"
#include "giao_integrals/nuclear.hpp"
#include "giao_integrals/overlap.hpp"
#include "giao_integrals/property.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <new>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

std::size_t allocation_count = 0;

void* operator new(std::size_t size) {
    ++allocation_count;
    if (void* memory = std::malloc(size)) {
        return memory;
    }
    throw std::bad_alloc();
}

void operator delete(void* memory) noexcept { std::free(memory); }

void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void check_close(giao::Complex actual, giao::Complex expected,
                 const std::string& message, double tolerance = 5.0e-13) {
    const double error = std::abs(actual - expected);
    if (error > tolerance * std::max(1.0, std::abs(expected))) {
        std::cerr << "FAIL: " << message << ": actual=" << actual
                  << " expected=" << expected << " error=" << error << '\n';
        ++failures;
    }
}

void test_cartesian_ordering() {
    const auto values = giao::cartesian_components(2);
    const std::vector<giao::CartesianExponent> expected{
        {2, 0, 0}, {1, 1, 0}, {1, 0, 1},
        {0, 2, 0}, {0, 1, 1}, {0, 0, 2}};
    check(values.size() == expected.size(), "d shell component count");
    for (std::size_t index = 0; index < expected.size(); ++index) {
        check(values[index].x == expected[index].x &&
                  values[index].y == expected[index].y &&
                  values[index].z == expected[index].z,
              "d shell component ordering");
    }
}

void test_normalization() {
    const double exponent = 0.73;
    const double expected =
        std::pow(2.0 * exponent / std::numbers::pi, 0.75);
    check(std::abs(giao::primitive_normalization(exponent, {}) - expected) <
              2.0e-15,
          "s primitive normalization");

    for (std::uint16_t angular_momentum = 0; angular_momentum <= 6;
         ++angular_momentum) {
        for (const auto component :
             giao::cartesian_components(angular_momentum)) {
            const giao::PrimitiveGaussian primitive(
                0.83, {0.2, -0.4, 0.7}, component);
            check_close(giao::primitive_overlap(primitive, primitive), {1.0, 0.0},
                        "normalized primitive self overlap", 8.0e-13);
        }
    }
}

void test_ss_and_product_sign() {
    const giao::PrimitiveGaussian bra(0.7, {-0.4, 0.2, 0.8});
    const giao::PrimitiveGaussian ket(1.3, {0.8, -0.5, 0.1});
    const giao::MagneticField field({0.3, -0.2, 0.5}, {9.0, -4.0, 2.0});
    const auto pair = giao::gaussian_product(bra, ket, field);
    check(std::abs(pair.exponent - 2.0) < 1.0e-15, "pair exponent");
    check(std::abs(pair.pair_wave_vector.x - 0.245) < 1.0e-15,
          "pair q x sign");
    check(std::abs(pair.pair_wave_vector.y - 0.405) < 1.0e-15,
          "pair q y sign");
    check(std::abs(pair.pair_wave_vector.z - 0.015) < 1.0e-15,
          "pair q z sign");

    const double gaussian = std::pow(std::numbers::pi / pair.exponent, 1.5);
    const auto expected =
        giao::primitive_normalization(bra.exponent, {}) *
        giao::primitive_normalization(ket.exponent, {}) * gaussian *
        pair.prefactor();
    check_close(giao::primitive_overlap(bra, ket, field), expected,
                "analytic finite-field ss overlap", 5.0e-14);
}

void test_hermiticity() {
    const giao::MagneticField field({0.19, -0.31, 0.27}, {0.4, -0.2, 0.1});
    for (std::uint16_t angular_a = 0; angular_a <= 4; ++angular_a) {
        for (const auto component_a : giao::cartesian_components(angular_a)) {
            const giao::PrimitiveGaussian bra(
                0.91, {-0.3, 0.2, 0.5}, component_a);
            for (std::uint16_t angular_b = 0; angular_b <= 4; ++angular_b) {
                for (const auto component_b :
                     giao::cartesian_components(angular_b)) {
                    const giao::PrimitiveGaussian ket(
                        1.17, {0.6, -0.4, 0.1}, component_b);
                    check_close(giao::primitive_overlap(bra, ket, field),
                                std::conj(giao::primitive_overlap(ket, bra, field)),
                                "primitive Hermiticity", 2.0e-12);
                }
            }
        }
    }
}

void test_shell_and_basis() {
    const giao::Shell shell(
        {0.1, -0.3, 0.2}, 3, {3.2, 0.9, 0.24}, {0.18, -0.42, 0.73});
    const giao::MagneticField field({0.2, 0.1, -0.3});
    std::vector<giao::Complex> block(giao::shell_pair_size(shell, shell));
    giao::IntegralWorkspace workspace;
    giao::compute_overlap(shell, shell, field, block, workspace);
    const auto allocations_before_reuse = allocation_count;
    giao::compute_overlap(shell, shell, field, block, workspace);
    const auto allocations_after_reuse = allocation_count;
    check(allocations_after_reuse == allocations_before_reuse,
          "warmed shell kernel performs no heap allocations");
    for (std::size_t row = 0; row < shell.ao_count(); ++row) {
        check_close(block[row * shell.ao_count() + row], {1.0, 0.0},
                    "normalized shell diagonal", 5.0e-13);
        for (std::size_t column = 0; column < shell.ao_count(); ++column) {
            check_close(block[row * shell.ao_count() + column],
                        std::conj(block[column * shell.ao_count() + row]),
                        "shell block Hermiticity", 8.0e-13);
        }
    }

    const giao::Shell general(
        {-0.5, 0.2, 0.3}, 1, {1.7, 0.4}, {0.5, 0.6, -0.2, 0.9}, 2);
    check(general.ao_count() == 6, "general contraction AO count");
    const giao::Basis basis({shell, general});
    check(basis.ao_count() == 16, "basis AO count");
    check(basis.ao_offsets()[0] == 0 && basis.ao_offsets()[1] == 10,
          "basis AO offsets");
    std::vector<giao::Complex> matrix(basis.ao_count() * basis.ao_count());
    giao::compute_overlap_matrix(basis, field, matrix);
    for (std::size_t row = 0; row < basis.ao_count(); ++row) {
        check_close(matrix[row * basis.ao_count() + row], {1.0, 0.0},
                    "basis overlap diagonal", 8.0e-13);
        for (std::size_t column = 0; column < basis.ao_count(); ++column) {
            check_close(matrix[row * basis.ao_count() + column],
                        std::conj(matrix[column * basis.ao_count() + row]),
                        "basis Hermiticity", 2.0e-12);
        }
    }

    bool rejected = false;
    try {
        std::vector<giao::Complex> wrong(block.size() - 1U);
        giao::compute_overlap(shell, shell, field, wrong, workspace);
    } catch (const std::length_error&) {
        rejected = true;
    }
    check(rejected, "shell driver rejects incorrect output size");
}

void test_validation() {
    bool rejected = false;
    try {
        [[maybe_unused]] const giao::PrimitiveGaussian bad(
            0.0, {0.0, 0.0, 0.0});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "non-positive exponent rejected");

    rejected = false;
    try {
        [[maybe_unused]] const giao::Shell bad({}, 0, {1.0}, {0.0});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "all-zero contraction rejected");
}

void test_one_electron_properties() {
    constexpr double exponent = 0.8;
    const giao::Vec3 center{0.4, -0.3, 0.7};
    const giao::PrimitiveGaussian primitive(exponent, center);
    giao::IntegralWorkspace workspace;
    check_close(giao::primitive_moment(
                    primitive, primitive,
                    giao::CartesianMoment(center, {2, 0, 0}), {}, workspace),
                {1.0 / (4.0 * exponent), 0.0}, "centered s x-square moment",
                5.0e-14);
    check_close(giao::primitive_kinetic(primitive, primitive, {}, workspace),
                {1.5 * exponent, 0.0}, "zero-field normalized s kinetic",
                5.0e-14);

    const giao::MagneticField field({0.2, -0.4, 0.7}, {-0.3, 0.1, 0.5});
    const auto wave_vector = field.london_wave_vector(center);
    const std::array<double, 3> wave{wave_vector.x, wave_vector.y, wave_vector.z};
    const std::array<giao::Axis, 3> axes{giao::Axis::x, giao::Axis::y,
                                         giao::Axis::z};
    for (std::size_t axis = 0; axis < 3; ++axis) {
        check_close(giao::primitive_gradient(primitive, primitive, axes[axis],
                                             field, workspace),
                    {0.0, -wave[axis]}, "London phase gradient expectation",
                    8.0e-14);
        check_close(giao::primitive_momentum(primitive, primitive, axes[axis],
                                             field, workspace),
                    {-wave[axis], 0.0}, "London phase momentum expectation",
                    8.0e-14);
    }
    const double field_squared =
        field.B.x * field.B.x + field.B.y * field.B.y + field.B.z * field.B.z;
    check_close(giao::primitive_magnetic_kinetic(primitive, primitive, field,
                                                 workspace),
                {1.5 * exponent + field_squared / (16.0 * exponent), 0.0},
                "gauge-independent magnetic s kinetic diagonal", 2.0e-13);

    const giao::PrimitiveGaussian other(1.1, {-0.2, 0.5, 0.1}, {1, 1, 0});
    check_close(giao::primitive_momentum(primitive, other, giao::Axis::z, field,
                                         workspace),
                giao::Complex{0.0, -1.0} *
                    giao::primitive_gradient(primitive, other, giao::Axis::z,
                                             field, workspace),
                "momentum equals -i gradient", 2.0e-13);
    check_close(giao::primitive_kinetic(primitive, other, field, workspace),
                std::conj(giao::primitive_kinetic(other, primitive, field,
                                                  workspace)),
                "canonical kinetic Hermiticity", 3.0e-13);
    check_close(giao::primitive_magnetic_kinetic(primitive, other, field,
                                                 workspace),
                std::conj(giao::primitive_magnetic_kinetic(
                    other, primitive, field, workspace)),
                "magnetic kinetic Hermiticity", 8.0e-13);
}

void test_boys_and_nuclear_attraction() {
    const auto at_zero = giao::boys_values({0.0, 0.0}, 8);
    for (std::size_t order = 0; order < at_zero.size(); ++order) {
        check_close(at_zero[order],
                    {1.0 / static_cast<double>(2U * order + 1U), 0.0},
                    "Boys value at zero", 2.0e-14);
    }

    const giao::Complex argument{-3.0, 2.0};
    const auto values = giao::boys_values(argument, 12);
    for (std::size_t order = 0; order < 12; ++order) {
        const auto residual =
            static_cast<double>(2U * order + 1U) * values[order] -
            std::exp(-argument) - 2.0 * argument * values[order + 1U];
        check(std::abs(residual) < 2.0e-12,
              "complex Boys recurrence residual");
    }

    const giao::PrimitiveGaussian bra(0.7, {-0.4, 0.2, 0.8});
    const giao::PrimitiveGaussian ket(1.3, {0.8, -0.5, 0.1});
    const giao::Nucleus nucleus(2.0, {0.1, -0.2, 0.3});
    const giao::MagneticField field({0.3, -0.2, 0.5}, {0.2, -0.1, 0.4});
    const std::array<giao::Nucleus, 1> nuclei{nucleus};
    const auto pair = giao::gaussian_product(bra, ket, field);
    const std::array<giao::Complex, 3> displacement{
        pair.complex_center[0] - nucleus.center.x,
        pair.complex_center[1] - nucleus.center.y,
        pair.complex_center[2] - nucleus.center.z};
    const auto boys_argument =
        pair.exponent * (displacement[0] * displacement[0] +
                         displacement[1] * displacement[1] +
                         displacement[2] * displacement[2]);
    const auto f0 = giao::boys_values(boys_argument, 0)[0];
    const auto expected =
        -nucleus.charge * (2.0 * std::numbers::pi / pair.exponent) *
        pair.prefactor() * giao::primitive_normalization(bra.exponent, {}) *
        giao::primitive_normalization(ket.exponent, {}) * f0;
    check_close(giao::primitive_nuclear_attraction(bra, ket, nuclei, field),
                expected, "analytic finite-field ss nuclear attraction",
                2.0e-13);

    const giao::PrimitiveGaussian higher(0.9, {-0.2, 0.5, 0.1}, {2, 1, 1});
    check_close(
        giao::primitive_nuclear_attraction(higher, ket, nuclei, field),
        std::conj(
            giao::primitive_nuclear_attraction(ket, higher, nuclei, field)),
        "nuclear-attraction Hermiticity", 8.0e-12);

    const giao::Shell shell({0.1, -0.3, 0.2}, 2, {1.8, 0.5}, {0.3, 0.8});
    std::vector<giao::Complex> block(giao::shell_pair_size(shell, shell));
    giao::NuclearAttractionWorkspace workspace;
    giao::compute_nuclear_attraction(shell, shell, nuclei, field, block,
                                     workspace);
    const auto allocations_before_reuse = allocation_count;
    giao::compute_nuclear_attraction(shell, shell, nuclei, field, block,
                                     workspace);
    check(allocation_count == allocations_before_reuse,
          "warmed nuclear-attraction shell kernel performs no heap allocations");
}

struct ConsumerState {
    std::size_t count{};
    giao::Complex value{};
};

struct CollectedBlocks {
    std::vector<giao::ShellQuartetIndex> quartets;
    std::vector<std::vector<giao::Complex>> values;
};

void collect_quartet(const giao::ShellQuartetBlockView& block, void* data) {
    auto& collected = *static_cast<CollectedBlocks*>(data);
    collected.quartets.push_back(block.shells);
    collected.values.emplace_back(block.values.begin(), block.values.end());
}

void capture_quartet(const giao::ShellQuartetBlockView& block, void* data) {
    auto& state = *static_cast<ConsumerState*>(data);
    ++state.count;
    state.value = block.values.front();
}

void test_electron_repulsion() {
    const giao::PrimitiveGaussian a(0.7, {-0.4, 0.2, 0.8});
    const giao::PrimitiveGaussian b(1.3, {0.8, -0.5, 0.1});
    const giao::PrimitiveGaussian c(0.9, {-0.2, 0.6, -0.3});
    const giao::PrimitiveGaussian d(1.1, {0.5, 0.1, -0.7});
    const giao::MagneticField field({0.3, -0.2, 0.5}, {0.2, -0.1, 0.4});
    const auto ab = giao::gaussian_product(a, b, field);
    const auto cd = giao::gaussian_product(c, d, field);
    const auto p = ab.exponent;
    const auto q = cd.exponent;
    const auto rho = p * q / (p + q);
    const auto dx = ab.complex_center[0] - cd.complex_center[0];
    const auto dy = ab.complex_center[1] - cd.complex_center[1];
    const auto dz = ab.complex_center[2] - cd.complex_center[2];
    const auto f0 = giao::boys_values(rho * (dx * dx + dy * dy + dz * dz), 0)[0];
    const auto expected =
        2.0 * std::pow(std::numbers::pi, 2.5) /
        (p * q * std::sqrt(p + q)) * ab.prefactor() * cd.prefactor() *
        giao::primitive_normalization(a.exponent, {}) *
        giao::primitive_normalization(b.exponent, {}) *
        giao::primitive_normalization(c.exponent, {}) *
        giao::primitive_normalization(d.exponent, {}) * f0;
    check_close(giao::primitive_eri(a, b, c, d, field), expected,
                "analytic finite-field ssss ERI", 3.0e-13);

    const giao::PrimitiveGaussian higher_a(0.8, {-0.3, 0.4, 0.1}, {2, 1, 0});
    const giao::PrimitiveGaussian higher_b(1.2, {0.5, -0.2, 0.7}, {0, 1, 1});
    const giao::PrimitiveGaussian higher_c(0.6, {0.2, 0.8, -0.4}, {1, 0, 1});
    const giao::PrimitiveGaussian higher_d(1.4, {-0.7, 0.1, 0.3}, {1, 1, 0});
    const auto value =
        giao::primitive_eri(higher_a, higher_b, higher_c, higher_d, field);
    check_close(value,
                giao::primitive_eri(higher_c, higher_d, higher_a, higher_b,
                                    field),
                "ERI pair exchange", 8.0e-12);
    check_close(value,
                std::conj(giao::primitive_eri(higher_b, higher_a, higher_d,
                                              higher_c, field)),
                "ERI conjugate double reversal", 8.0e-12);
    check(std::abs(value - giao::primitive_eri(higher_b, higher_a, higher_c,
                                               higher_d, field)) > 1.0e-8,
          "finite-field one-pair swap is not treated as a symmetry");

    const giao::Shell shell_a({0.1, -0.3, 0.2}, 1, {1.8, 0.5}, {0.3, 0.8});
    const giao::Shell shell_b({-0.4, 0.2, 0.6}, 0, {0.7}, {1.0});
    std::vector<giao::Complex> block(
        giao::shell_quartet_size(shell_a, shell_b, shell_a, shell_b));
    giao::EriWorkspace workspace;
    giao::compute_eri(shell_a, shell_b, shell_a, shell_b, field, block,
                      workspace);
    const auto allocations_before_reuse = allocation_count;
    giao::compute_eri(shell_a, shell_b, shell_a, shell_b, field, block,
                      workspace);
    check(allocation_count == allocations_before_reuse,
          "warmed ERI shell kernel performs no heap allocations");

    const giao::Basis basis({shell_b});
    const std::array<giao::ShellQuartetIndex, 1> quartets{{{0, 0, 0, 0}}};
    ConsumerState state;
    giao::for_each_eri_shell_quartet(basis, quartets, field, capture_quartet,
                                     &state);
    check(state.count == 1, "C++ ERI consumer receives requested quartet");
    std::array<giao::Complex, 1> tensor{};
    giao::compute_eri_tensor(basis, field, tensor);
    check_close(tensor[0], state.value, "full ERI tensor matches consumer");

    const auto canonical =
        giao::canonicalize_shell_quartet({3, 2, 1, 0});
    check(canonical.shells.as_array() == std::array<std::uint32_t, 4>{0, 1, 2, 3} &&
              canonical.conjugate,
          "shell-quartet canonicalization records conjugation");
}

void test_eri_screening_and_parallel_driver() {
    const giao::Shell near({0.0, 0.0, 0.0}, 0, {0.8}, {1.0});
    const giao::Shell far({8.0, 0.0, 0.0}, 0, {0.8}, {1.0});
    const giao::Basis basis({near, far});
    const giao::MagneticField field({0.1, -0.2, 0.3});
    const giao::EriSchwarzBounds bounds(basis, field);
    check(bounds.shell_count() == 2U, "ERI Schwarz shell count");
    check(std::abs(bounds(0, 1) - bounds(1, 0)) < 1.0e-30,
          "ERI Schwarz factors are symmetric");

    std::vector<giao::ShellQuartetIndex> quartets;
    for (std::uint32_t a = 0; a < 2; ++a) {
        for (std::uint32_t b = 0; b < 2; ++b) {
            for (std::uint32_t c = 0; c < 2; ++c) {
                for (std::uint32_t d = 0; d < 2; ++d) {
                    quartets.push_back({a, b, c, d});
                    std::array<giao::Complex, 1> value{};
                    giao::EriWorkspace workspace;
                    giao::compute_eri(basis.shells()[a], basis.shells()[b],
                                      basis.shells()[c], basis.shells()[d],
                                      field, value, workspace);
                    const double schwarz = bounds(a, b) * bounds(c, d);
                    check(std::abs(value[0]) <=
                              schwarz * (1.0 + 2.0e-12) + 1.0e-15,
                          "complex shell ERI obeys Schwarz bound");
                }
            }
        }
    }

    CollectedBlocks unscreened;
    const auto unscreened_statistics = giao::evaluate_eri_shell_quartets(
        basis, quartets, field, {}, nullptr, collect_quartet, &unscreened);
    check(unscreened_statistics.requested_quartets == quartets.size() &&
              unscreened_statistics.computed_quartets == quartets.size() &&
              unscreened_statistics.screened_quartets == 0U,
          "zero threshold preserves the unscreened ERI path");

    CollectedBlocks screened;
    const auto screened_statistics = giao::evaluate_eri_shell_quartets(
        basis, quartets, field, {1.0e-8, 1U}, &bounds, collect_quartet,
        &screened);
    check(screened_statistics.screened_quartets > 0U &&
              screened_statistics.computed_quartets +
                      screened_statistics.screened_quartets ==
                  quartets.size(),
          "positive threshold screens bounded shell quartets");

    if (giao::openmp_enabled()) {
        CollectedBlocks parallel;
        const auto parallel_statistics = giao::evaluate_eri_shell_quartets(
            basis, quartets, field, {0.0, 2U}, nullptr, collect_quartet,
            &parallel);
        check(parallel_statistics.computed_quartets == quartets.size(),
              "OpenMP ERI driver computes every unscreened quartet");
        check(parallel.quartets.size() == unscreened.quartets.size(),
              "OpenMP ERI callback count");
        for (std::size_t index = 0; index < parallel.values.size(); ++index) {
            check(parallel.quartets[index].as_array() ==
                      unscreened.quartets[index].as_array(),
                  "OpenMP ERI callback ordering is deterministic");
            check_close(parallel.values[index][0], unscreened.values[index][0],
                        "OpenMP ERI block is bitwise-path equivalent", 0.0);
        }
    }
}

void test_analytic_derivatives() {
    const giao::PrimitiveGaussian bra(0.73, {0.2, -0.4, 0.1}, {1, 0, 1});
    const giao::PrimitiveGaussian ket(1.11, {-0.3, 0.5, -0.2}, {0, 1, 0});
    const giao::MagneticField field({0.17, -0.11, 0.23}, {0.2, -0.3, 0.1});
    constexpr double step = 1.0e-5;

    const auto center = giao::primitive_overlap_center_derivatives(bra, ket, field);
    const giao::PrimitiveGaussian plus_center(
        bra.exponent, {bra.center.x + step, bra.center.y, bra.center.z}, bra.angular);
    const giao::PrimitiveGaussian minus_center(
        bra.exponent, {bra.center.x - step, bra.center.y, bra.center.z}, bra.angular);
    const auto finite_center = (giao::primitive_overlap(plus_center, ket, field) -
                                giao::primitive_overlap(minus_center, ket, field)) /
                               (2.0 * step);
    check_close(center[0], finite_center, "analytic overlap center derivative", 2.0e-9);

    const auto magnetic = giao::primitive_overlap_magnetic_derivatives(bra, ket, field);
    const giao::MagneticField plus_field({field.B.x, field.B.y + step, field.B.z},
                                         field.gauge_origin);
    const giao::MagneticField minus_field({field.B.x, field.B.y - step, field.B.z},
                                          field.gauge_origin);
    const auto finite_magnetic = (giao::primitive_overlap(bra, ket, plus_field) -
                                  giao::primitive_overlap(bra, ket, minus_field)) /
                                 (2.0 * step);
    check_close(magnetic[1], finite_magnetic, "analytic overlap magnetic derivative",
                2.0e-9);

    const std::array<giao::Nucleus, 1> nucleus{giao::Nucleus(1.4, {0.1, -0.2, 0.3})};
    std::array<giao::Complex, 3> potential_derivative{};
    giao::primitive_nuclear_attraction_nucleus_derivatives(bra, ket, nucleus, field,
                                                           potential_derivative);
    const std::array<giao::Nucleus, 1> plus_nucleus{
        giao::Nucleus(1.4, {0.1, -0.2, 0.3 + step})};
    const std::array<giao::Nucleus, 1> minus_nucleus{
        giao::Nucleus(1.4, {0.1, -0.2, 0.3 - step})};
    const auto finite_potential =
        (giao::primitive_nuclear_attraction(bra, ket, plus_nucleus, field) -
         giao::primitive_nuclear_attraction(bra, ket, minus_nucleus, field)) /
        (2.0 * step);
    check_close(potential_derivative[2], finite_potential,
                "analytic attraction potential-center derivative", 3.0e-9);

    const auto physical_magnetic =
        giao::primitive_magnetic_kinetic_magnetic_derivatives(bra, ket, field);
    giao::IntegralWorkspace physical_workspace;
    const auto finite_physical =
        (giao::primitive_magnetic_kinetic(bra, ket, plus_field, physical_workspace) -
         giao::primitive_magnetic_kinetic(bra, ket, minus_field, physical_workspace)) /
        (2.0 * step);
    check_close(physical_magnetic[1], finite_physical,
                "physical magnetic kinetic field derivative", 3.0e-9);

    const giao::PrimitiveGaussian c(0.8, {0.4, 0.2, -0.1}, {0, 0, 0});
    const giao::PrimitiveGaussian d(1.2, {-0.2, -0.4, 0.3}, {0, 0, 0});
    const auto eri_center =
        giao::primitive_eri_center_derivatives(bra, ket, c, d, field);
    const auto finite_eri = (giao::primitive_eri(plus_center, ket, c, d, field) -
                             giao::primitive_eri(minus_center, ket, c, d, field)) /
                            (2.0 * step);
    check_close(eri_center[0], finite_eri, "analytic ERI center derivative", 3.0e-9);

    const giao::Shell shell_a({0.1, -0.2, 0.3}, 0, {0.7}, {1.0});
    const giao::Shell shell_b({-0.4, 0.2, 0.1}, 1, {0.9}, {1.0});
    std::vector<giao::Complex> shell_derivatives(
        6U * giao::shell_pair_size(shell_a, shell_b));
    giao::compute_overlap_center_derivatives(shell_a, shell_b, field,
                                             shell_derivatives);
    check(shell_derivatives.size() == 18U,
          "derivative shell block has leading center and axis dimensions");
}

}  // namespace

int main() {
    test_cartesian_ordering();
    test_normalization();
    test_ss_and_product_sign();
    test_hermiticity();
    test_shell_and_basis();
    test_validation();
    test_one_electron_properties();
    test_boys_and_nuclear_attraction();
    test_electron_repulsion();
    test_eri_screening_and_parallel_driver();
    test_analytic_derivatives();
    if (failures != 0) {
        std::cerr << failures << " test checks failed\n";
        return 1;
    }
    std::cout << "all C++ integral checks passed\n";
    return 0;
}

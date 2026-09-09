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

}  // namespace

int main() {
    test_cartesian_ordering();
    test_normalization();
    test_ss_and_product_sign();
    test_hermiticity();
    test_shell_and_basis();
    test_validation();
    test_one_electron_properties();
    if (failures != 0) {
        std::cerr << failures << " test checks failed\n";
        return 1;
    }
    std::cout << "all C++ integral checks passed\n";
    return 0;
}

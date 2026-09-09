#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace giao {

using Complex = std::complex<double>;

struct Vec3 {
    double x{};
    double y{};
    double z{};

    [[nodiscard]] constexpr double operator[](std::size_t axis) const {
        return axis == 0 ? x : (axis == 1 ? y : z);
    }
};

struct CartesianExponent {
    std::uint16_t x{};
    std::uint16_t y{};
    std::uint16_t z{};

    [[nodiscard]] constexpr std::uint32_t total() const noexcept {
        return static_cast<std::uint32_t>(x) + static_cast<std::uint32_t>(y) +
               static_cast<std::uint32_t>(z);
    }

    [[nodiscard]] constexpr std::uint16_t operator[](std::size_t axis) const {
        return axis == 0 ? x : (axis == 1 ? y : z);
    }
};

struct PrimitiveGaussian {
    double exponent{};
    double coefficient{1.0};
    Vec3 center{};
    CartesianExponent angular{};
    bool normalized{true};

    PrimitiveGaussian(double exponent_in, Vec3 center_in,
                      CartesianExponent angular_in = {},
                      double coefficient_in = 1.0,
                      bool normalized_in = true);
};

enum class ContractionNormalization {
    normalize,
    as_provided,
};

class Shell {
  public:
    Shell(Vec3 center, std::uint16_t angular_momentum,
          std::vector<double> exponents, std::vector<double> coefficients,
          std::size_t contraction_count = 1,
          ContractionNormalization normalization =
              ContractionNormalization::normalize);

    [[nodiscard]] Vec3 center() const noexcept { return center_; }
    [[nodiscard]] std::uint16_t angular_momentum() const noexcept {
        return angular_momentum_;
    }
    [[nodiscard]] std::size_t primitive_count() const noexcept {
        return exponents_.size();
    }
    [[nodiscard]] std::size_t contraction_count() const noexcept {
        return contraction_count_;
    }
    [[nodiscard]] std::size_t cartesian_count() const noexcept {
        return components_.size();
    }
    [[nodiscard]] std::size_t ao_count() const noexcept {
        return contraction_count_ * cartesian_count();
    }
    [[nodiscard]] ContractionNormalization normalization_policy() const noexcept {
        return normalization_;
    }
    [[nodiscard]] std::span<const double> exponents() const noexcept {
        return exponents_;
    }
    [[nodiscard]] std::span<const double> coefficients() const noexcept {
        return coefficients_;
    }
    [[nodiscard]] std::span<const CartesianExponent> components() const noexcept {
        return components_;
    }
    [[nodiscard]] double coefficient(std::size_t contraction,
                                     std::size_t primitive) const noexcept {
        return coefficients_[contraction * primitive_count() + primitive];
    }

  private:
    Vec3 center_{};
    std::uint16_t angular_momentum_{};
    std::vector<double> exponents_;
    std::vector<double> coefficients_;
    std::size_t contraction_count_{};
    ContractionNormalization normalization_{};
    std::vector<CartesianExponent> components_;
};

class Basis {
  public:
    explicit Basis(std::vector<Shell> shells);

    [[nodiscard]] std::span<const Shell> shells() const noexcept {
        return shells_;
    }
    [[nodiscard]] std::span<const std::size_t> ao_offsets() const noexcept {
        return ao_offsets_;
    }
    [[nodiscard]] std::size_t ao_count() const noexcept { return ao_count_; }

  private:
    std::vector<Shell> shells_;
    std::vector<std::size_t> ao_offsets_;
    std::size_t ao_count_{};
};

struct MagneticField {
    Vec3 B{};
    Vec3 gauge_origin{};

    MagneticField(Vec3 field = {}, Vec3 origin = {});
    [[nodiscard]] Vec3 london_wave_vector(Vec3 center) const noexcept;
};

struct Nucleus {
    double charge{};
    Vec3 center{};

    Nucleus(double charge_in, Vec3 center_in);
};

[[nodiscard]] double primitive_normalization(
    double exponent, CartesianExponent angular);

[[nodiscard]] std::vector<CartesianExponent> cartesian_components(
    std::uint16_t total_angular_momentum);

}  // namespace giao

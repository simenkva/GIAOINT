#pragma once

#include "giao_integrals/types.hpp"

#include <span>
#include <vector>

namespace giao {

struct GaussianPairData {
    double exponent{};
    double reduced_exponent{};
    Vec3 product_center{};
    Vec3 pair_wave_vector{};
    std::array<Complex, 3> complex_center{};
    double ordinary_prefactor{};
    Complex london_prefactor{};

    [[nodiscard]] Complex prefactor() const noexcept {
        return ordinary_prefactor * london_prefactor;
    }
};

class IntegralWorkspace {
  public:
    void prepare_overlap(std::size_t maximum_hermite_order,
                         std::size_t normalization_count_a = 0,
                         std::size_t normalization_count_b = 0);

  private:
    friend Complex primitive_overlap(const PrimitiveGaussian&,
                                     const PrimitiveGaussian&,
                                     const MagneticField&,
                                     IntegralWorkspace&);
    friend void compute_overlap(const Shell&, const Shell&,
                                const MagneticField&, std::span<Complex>,
                                IntegralWorkspace&);

    std::vector<double> hermite_current_;
    std::vector<double> hermite_next_;
    std::vector<double> normalizations_a_;
    std::vector<double> normalizations_b_;
};

[[nodiscard]] GaussianPairData gaussian_product(
    const PrimitiveGaussian& bra, const PrimitiveGaussian& ket,
    const MagneticField& field = {});

[[nodiscard]] Complex primitive_overlap(
    const PrimitiveGaussian& bra, const PrimitiveGaussian& ket,
    const MagneticField& field, IntegralWorkspace& workspace);

[[nodiscard]] Complex primitive_overlap(
    const PrimitiveGaussian& bra, const PrimitiveGaussian& ket,
    const MagneticField& field = {});

[[nodiscard]] std::size_t shell_pair_size(const Shell& a,
                                          const Shell& b) noexcept;

void compute_overlap(const Shell& a, const Shell& b,
                     const MagneticField& field, std::span<Complex> output,
                     IntegralWorkspace& workspace);

void compute_overlap_matrix(const Basis& basis, const MagneticField& field,
                            std::span<Complex> output);

}  // namespace giao

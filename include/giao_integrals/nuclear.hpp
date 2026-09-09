#pragma once

#include "giao_integrals/boys.hpp"
#include "giao_integrals/overlap.hpp"
#include "giao_integrals/types.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace giao {

namespace detail {
struct NuclearKernel;
}

class NuclearAttractionWorkspace {
  public:
    void prepare(std::size_t maximum_angular_order,
                 std::size_t normalization_count_a = 0,
                 std::size_t normalization_count_b = 0);

  private:
    friend struct detail::NuclearKernel;
    friend Complex primitive_nuclear_attraction(
        const PrimitiveGaussian&, const PrimitiveGaussian&,
        std::span<const Nucleus>, const MagneticField&,
        NuclearAttractionWorkspace&);
    friend void compute_nuclear_attraction(
        const Shell&, const Shell&, std::span<const Nucleus>,
        const MagneticField&, std::span<Complex>,
        NuclearAttractionWorkspace&);

    std::vector<double> hermite_current_;
    std::vector<double> hermite_next_;
    std::array<std::vector<double>, 3> hermite_coefficients_;
    std::vector<double> normalizations_a_;
    std::vector<double> normalizations_b_;
    std::vector<Complex> boys_;
    std::vector<Complex> coulomb_;
    std::vector<unsigned char> coulomb_ready_;
};

[[nodiscard]] Complex primitive_nuclear_attraction(
    const PrimitiveGaussian& bra, const PrimitiveGaussian& ket,
    std::span<const Nucleus> nuclei, const MagneticField& field,
    NuclearAttractionWorkspace& workspace);

[[nodiscard]] Complex primitive_nuclear_attraction(
    const PrimitiveGaussian& bra, const PrimitiveGaussian& ket,
    std::span<const Nucleus> nuclei, const MagneticField& field = {});

void compute_nuclear_attraction(
    const Shell& a, const Shell& b, std::span<const Nucleus> nuclei,
    const MagneticField& field, std::span<Complex> output,
    NuclearAttractionWorkspace& workspace);

void compute_nuclear_attraction_matrix(
    const Basis& basis, std::span<const Nucleus> nuclei,
    const MagneticField& field, std::span<Complex> output);

}  // namespace giao

#pragma once

#include "giao_integrals/boys.hpp"
#include "giao_integrals/overlap.hpp"
#include "giao_integrals/types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace giao {

namespace detail {
struct EriKernel;
}

class EriWorkspace {
  public:
    void prepare(std::array<std::size_t, 7> auxiliary_dimensions,
                 std::size_t maximum_hermite_order,
                 std::array<std::size_t, 4> normalization_counts = {});

  private:
    friend struct detail::EriKernel;
    friend void compute_eri(const Shell&, const Shell&, const Shell&,
                            const Shell&, const MagneticField&,
                            std::span<Complex>, EriWorkspace&);

    std::vector<double> hermite_current_;
    std::vector<double> hermite_next_;
    std::array<std::vector<double>, 6> hermite_coefficients_;
    std::array<std::vector<double>, 4> normalizations_;
    std::vector<Complex> boys_;
    std::vector<Complex> auxiliary_;
    std::vector<unsigned char> auxiliary_ready_;
};

struct ShellQuartetIndex {
    std::uint32_t a{};
    std::uint32_t b{};
    std::uint32_t c{};
    std::uint32_t d{};

    [[nodiscard]] constexpr auto as_array() const noexcept {
        return std::array<std::uint32_t, 4>{a, b, c, d};
    }
};

struct CanonicalShellQuartet {
    ShellQuartetIndex shells{};
    bool conjugate{};
};

struct ShellQuartetBlockView {
    ShellQuartetIndex shells{};
    std::array<std::size_t, 4> shape{};
    std::span<const Complex> values{};
};

using QuartetConsumer = void (*)(const ShellQuartetBlockView&, void* user_data);

[[nodiscard]] CanonicalShellQuartet canonicalize_shell_quartet(
    ShellQuartetIndex quartet) noexcept;

[[nodiscard]] std::size_t shell_quartet_size(
    const Shell& a, const Shell& b, const Shell& c,
    const Shell& d);

[[nodiscard]] Complex primitive_eri(
    const PrimitiveGaussian& a, const PrimitiveGaussian& b,
    const PrimitiveGaussian& c, const PrimitiveGaussian& d,
    const MagneticField& field, EriWorkspace& workspace);

[[nodiscard]] Complex primitive_eri(
    const PrimitiveGaussian& a, const PrimitiveGaussian& b,
    const PrimitiveGaussian& c, const PrimitiveGaussian& d,
    const MagneticField& field = {});

void compute_eri(const Shell& a, const Shell& b, const Shell& c,
                 const Shell& d, const MagneticField& field,
                 std::span<Complex> output, EriWorkspace& workspace);

void for_each_eri_shell_quartet(
    const Basis& basis, std::span<const ShellQuartetIndex> quartets,
    const MagneticField& field, QuartetConsumer consumer, void* user_data);

void compute_eri_tensor(const Basis& basis, const MagneticField& field,
                        std::span<Complex> output);

}  // namespace giao

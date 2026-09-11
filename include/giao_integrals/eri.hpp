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
    std::array<std::vector<double>, 2> pair_hermite_products_;
    std::array<bool, 2> hermite_cache_valid_{};
    std::array<std::array<std::uint16_t, 6>, 2> hermite_angular_key_{};
    std::array<std::array<double, 10>, 2> hermite_geometry_key_{};
    std::array<std::vector<double>, 4> normalizations_;
    std::vector<Complex> boys_;
    bool boys_cache_valid_{};
    Complex boys_argument_{};
    BoysScaling boys_scaling_{BoysScaling::unscaled};
    std::size_t boys_maximum_order_{};
    std::vector<Complex> auxiliary_;
    std::vector<double> real_auxiliary_;
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

struct EriEvaluationOptions {
    double screening_threshold{};
    std::size_t thread_count{1U};
};

struct EriStatistics {
    std::size_t requested_quartets{};
    std::size_t computed_quartets{};
    std::size_t screened_quartets{};
};

class EriSchwarzBounds {
  public:
    explicit EriSchwarzBounds(const Basis& basis,
                              const MagneticField& field = {});

    [[nodiscard]] std::size_t shell_count() const noexcept {
        return shell_count_;
    }
    [[nodiscard]] double operator()(std::size_t a,
                                    std::size_t b) const;
    [[nodiscard]] std::span<const double> values() const noexcept {
        return values_;
    }
    [[nodiscard]] bool matches(const Basis& basis,
                               const MagneticField& field) const noexcept;

  private:
    const Basis* basis_identity_{};
    MagneticField field_{};
    std::size_t shell_count_{};
    std::vector<double> values_;
};

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

[[nodiscard]] EriStatistics evaluate_eri_shell_quartets(
    const Basis& basis, std::span<const ShellQuartetIndex> quartets,
    const MagneticField& field, const EriEvaluationOptions& options,
    const EriSchwarzBounds* bounds, QuartetConsumer consumer,
    void* user_data);

void compute_eri_tensor(const Basis& basis, const MagneticField& field,
                        std::span<Complex> output);

[[nodiscard]] EriStatistics compute_eri_tensor(
    const Basis& basis, const MagneticField& field,
    const EriEvaluationOptions& options, const EriSchwarzBounds* bounds,
    std::span<Complex> output);

[[nodiscard]] bool openmp_enabled() noexcept;
[[nodiscard]] std::size_t openmp_max_threads() noexcept;

}  // namespace giao

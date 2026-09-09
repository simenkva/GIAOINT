#pragma once

#include "giao_integrals/overlap.hpp"

#include <cstdint>
#include <span>

namespace giao {

enum class Axis : std::uint8_t { x = 0, y = 1, z = 2 };

struct CartesianMoment {
    Vec3 origin{};
    CartesianExponent powers{};

    CartesianMoment(Vec3 origin_in = {}, CartesianExponent powers_in = {});
};

[[nodiscard]] Complex primitive_moment(
    const PrimitiveGaussian& bra, const PrimitiveGaussian& ket,
    const CartesianMoment& moment, const MagneticField& field,
    IntegralWorkspace& workspace);

[[nodiscard]] Complex primitive_gradient(
    const PrimitiveGaussian& bra, const PrimitiveGaussian& ket, Axis axis,
    const MagneticField& field, IntegralWorkspace& workspace);

[[nodiscard]] Complex primitive_momentum(
    const PrimitiveGaussian& bra, const PrimitiveGaussian& ket, Axis axis,
    const MagneticField& field, IntegralWorkspace& workspace);

[[nodiscard]] Complex primitive_kinetic(
    const PrimitiveGaussian& bra, const PrimitiveGaussian& ket,
    const MagneticField& field, IntegralWorkspace& workspace);

[[nodiscard]] Complex primitive_magnetic_kinetic(
    const PrimitiveGaussian& bra, const PrimitiveGaussian& ket,
    const MagneticField& field, IntegralWorkspace& workspace);

void compute_moment(const Shell& a, const Shell& b,
                    const CartesianMoment& moment,
                    const MagneticField& field, std::span<Complex> output,
                    IntegralWorkspace& workspace);

void compute_gradient(const Shell& a, const Shell& b, Axis axis,
                      const MagneticField& field, std::span<Complex> output,
                      IntegralWorkspace& workspace);

void compute_momentum(const Shell& a, const Shell& b, Axis axis,
                      const MagneticField& field, std::span<Complex> output,
                      IntegralWorkspace& workspace);

void compute_kinetic(const Shell& a, const Shell& b,
                     const MagneticField& field, std::span<Complex> output,
                     IntegralWorkspace& workspace);

void compute_magnetic_kinetic(const Shell& a, const Shell& b,
                              const MagneticField& field,
                              std::span<Complex> output,
                              IntegralWorkspace& workspace);

void compute_moment_matrix(const Basis& basis, const CartesianMoment& moment,
                           const MagneticField& field,
                           std::span<Complex> output);

void compute_gradient_matrix(const Basis& basis, Axis axis,
                             const MagneticField& field,
                             std::span<Complex> output);

void compute_momentum_matrix(const Basis& basis, Axis axis,
                             const MagneticField& field,
                             std::span<Complex> output);

void compute_kinetic_matrix(const Basis& basis, const MagneticField& field,
                            std::span<Complex> output);

void compute_magnetic_kinetic_matrix(const Basis& basis,
                                     const MagneticField& field,
                                     std::span<Complex> output);

}  // namespace giao

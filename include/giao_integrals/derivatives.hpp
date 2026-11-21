#pragma once

#include "giao_integrals/eri.hpp"
#include "giao_integrals/nuclear.hpp"
#include "giao_integrals/property.hpp"

#include <array>
#include <span>

namespace giao {

// Derivative buffers are row-major with the derivative indices leading:
// pair-center [2,3,na,nb], magnetic [3,na,nb], nuclear potential
// [nnucleus,3,na,nb], and quartet-center [4,3,na,nb,nc,nd].

// Literal derivatives of <a|(r-origin)^powers|b>, including London response.
[[nodiscard]] std::array<Complex, 6> primitive_moment_center_derivatives(
    const PrimitiveGaussian& bra, const PrimitiveGaussian& ket,
    const CartesianMoment& moment, const MagneticField& field = {});
[[nodiscard]] std::array<Complex, 3> primitive_moment_origin_derivatives(
    const PrimitiveGaussian& bra, const PrimitiveGaussian& ket,
    const CartesianMoment& moment, const MagneticField& field = {});
void compute_moment_center_derivatives(const Shell& a, const Shell& b,
    const CartesianMoment& moment, const MagneticField& field, std::span<Complex> output);
void compute_moment_origin_derivatives(const Shell& a, const Shell& b,
    const CartesianMoment& moment, const MagneticField& field, std::span<Complex> output);
// shell_output [nshell,3,nao,nao]; origin_output [3,nao,nao].
void compute_moment_nuclear_derivative_matrices(const Basis& basis,
    const CartesianMoment& moment, const MagneticField& field,
    std::span<Complex> shell_output, std::span<Complex> origin_output,
    std::size_t threads = 1U);

[[nodiscard]] std::array<Complex, 6>
primitive_overlap_center_derivatives(const PrimitiveGaussian& bra,
                                     const PrimitiveGaussian& ket,
                                     const MagneticField& field = {});
[[nodiscard]] std::array<Complex, 3>
primitive_overlap_magnetic_derivatives(const PrimitiveGaussian& bra,
                                       const PrimitiveGaussian& ket,
                                       const MagneticField& field = {});

[[nodiscard]] std::array<Complex, 6>
primitive_kinetic_center_derivatives(const PrimitiveGaussian& bra,
                                     const PrimitiveGaussian& ket,
                                     const MagneticField& field = {});
[[nodiscard]] std::array<Complex, 3>
primitive_kinetic_magnetic_derivatives(const PrimitiveGaussian& bra,
                                       const PrimitiveGaussian& ket,
                                       const MagneticField& field = {});

[[nodiscard]] std::array<Complex, 6>
primitive_magnetic_kinetic_center_derivatives(const PrimitiveGaussian& bra,
                                              const PrimitiveGaussian& ket,
                                              const MagneticField& field = {});
[[nodiscard]] std::array<Complex, 3>
primitive_magnetic_kinetic_magnetic_derivatives(const PrimitiveGaussian& bra,
                                                const PrimitiveGaussian& ket,
                                                const MagneticField& field = {});

[[nodiscard]] std::array<Complex, 6> primitive_nuclear_attraction_center_derivatives(
    const PrimitiveGaussian& bra, const PrimitiveGaussian& ket,
    std::span<const Nucleus> nuclei, const MagneticField& field = {});
[[nodiscard]] std::array<Complex, 3> primitive_nuclear_attraction_magnetic_derivatives(
    const PrimitiveGaussian& bra, const PrimitiveGaussian& ket,
    std::span<const Nucleus> nuclei, const MagneticField& field = {});
void primitive_nuclear_attraction_nucleus_derivatives(const PrimitiveGaussian& bra,
                                                      const PrimitiveGaussian& ket,
                                                      std::span<const Nucleus> nuclei,
                                                      const MagneticField& field,
                                                      std::span<Complex> output);

[[nodiscard]] std::array<Complex, 12>
primitive_eri_center_derivatives(const PrimitiveGaussian& a, const PrimitiveGaussian& b,
                                 const PrimitiveGaussian& c, const PrimitiveGaussian& d,
                                 const MagneticField& field = {});
[[nodiscard]] std::array<Complex, 3> primitive_eri_magnetic_derivatives(
    const PrimitiveGaussian& a, const PrimitiveGaussian& b, const PrimitiveGaussian& c,
    const PrimitiveGaussian& d, const MagneticField& field = {});

void compute_overlap_center_derivatives(const Shell& a, const Shell& b,
                                        const MagneticField& field,
                                        std::span<Complex> output);
void compute_overlap_magnetic_derivatives(const Shell& a, const Shell& b,
                                          const MagneticField& field,
                                          std::span<Complex> output);
void compute_kinetic_center_derivatives(const Shell& a, const Shell& b,
                                        const MagneticField& field,
                                        std::span<Complex> output);
void compute_kinetic_magnetic_derivatives(const Shell& a, const Shell& b,
                                          const MagneticField& field,
                                          std::span<Complex> output);
void compute_magnetic_kinetic_center_derivatives(const Shell& a, const Shell& b,
                                                 const MagneticField& field,
                                                 std::span<Complex> output);
void compute_magnetic_kinetic_magnetic_derivatives(const Shell& a, const Shell& b,
                                                   const MagneticField& field,
                                                   std::span<Complex> output);
void compute_nuclear_attraction_center_derivatives(const Shell& a, const Shell& b,
                                                   std::span<const Nucleus> nuclei,
                                                   const MagneticField& field,
                                                   std::span<Complex> output);
void compute_nuclear_attraction_magnetic_derivatives(const Shell& a, const Shell& b,
                                                     std::span<const Nucleus> nuclei,
                                                     const MagneticField& field,
                                                     std::span<Complex> output);
void compute_nuclear_attraction_nucleus_derivatives(const Shell& a, const Shell& b,
                                                    std::span<const Nucleus> nuclei,
                                                    const MagneticField& field,
                                                    std::span<Complex> output);

void compute_eri_center_derivatives(const Shell& a, const Shell& b, const Shell& c,
                                    const Shell& d, const MagneticField& field,
                                    std::span<Complex> output);
void compute_eri_magnetic_derivatives(const Shell& a, const Shell& b, const Shell& c,
                                      const Shell& d, const MagneticField& field,
                                      std::span<Complex> output);

void compute_overlap_nuclear_derivative_matrices(const Basis& basis,
                                                 const MagneticField& field,
                                                 std::span<Complex> output);
void compute_overlap_magnetic_derivative_matrices(const Basis& basis,
                                                  const MagneticField& field,
                                                  std::span<Complex> output);
void compute_kinetic_nuclear_derivative_matrices(const Basis& basis,
                                                 const MagneticField& field,
                                                 std::span<Complex> output);
void compute_kinetic_magnetic_derivative_matrices(const Basis& basis,
                                                  const MagneticField& field,
                                                  std::span<Complex> output);
void compute_magnetic_kinetic_nuclear_derivative_matrices(const Basis& basis,
                                                          const MagneticField& field,
                                                          std::span<Complex> output);
void compute_magnetic_kinetic_magnetic_derivative_matrices(const Basis& basis,
                                                           const MagneticField& field,
                                                           std::span<Complex> output);
void compute_nuclear_attraction_nuclear_derivative_matrices(
    const Basis& basis, std::span<const Nucleus> nuclei, const MagneticField& field,
    std::span<Complex> shell_center_output, std::span<Complex> nucleus_center_output);
void compute_nuclear_attraction_magnetic_derivative_matrices(
    const Basis& basis, std::span<const Nucleus> nuclei, const MagneticField& field,
    std::span<Complex> output);

} // namespace giao

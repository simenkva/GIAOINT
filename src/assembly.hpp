#pragma once
#include "giao_integrals/derivatives.hpp"

namespace giao::detail {
enum class AtomOperator {overlap,kinetic,magnetic_kinetic,attraction,moment};
void atom_one_electron_derivatives(const Basis& basis,std::span<const std::size_t> atoms,
    std::size_t natom,std::span<const Nucleus> nuclei,const MagneticField& field,
    AtomOperator op,const CartesianMoment& moment,bool mixed,std::span<Complex> output,
    std::span<Complex> origin,std::size_t threads);
struct DensityPair {std::span<const Complex> p,q;};
void contract_eri_derivatives(const Basis& basis,std::span<const std::size_t> atoms,
    std::size_t natom,std::span<const ShellQuartetIndex> quartets,
    std::span<const std::size_t> offsets,std::span<const Complex> values,
    std::span<const DensityPair> coulomb,std::span<const DensityPair> exchange,
    std::span<const Complex> gamma,std::span<const double> real_gamma,
    std::span<Complex> output,std::size_t threads);
} // namespace giao::detail

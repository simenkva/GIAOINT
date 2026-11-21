#include "assembly.hpp"
#include "parallel.hpp"
#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

namespace giao::detail {
namespace {
void validate_atoms(const Basis& basis,std::span<const std::size_t> atoms,std::size_t natom) {
    if (atoms.size()!=basis.shells().size()) {throw std::invalid_argument("shell_atoms must contain one atom per shell");}
    for (const auto atom:atoms) {if (atom>=natom) {throw std::out_of_range("shell atom index is out of range");}}
}
void pair_response(const Shell& a,const Shell& b,std::span<const Nucleus> nuclei,
    const MagneticField& field,AtomOperator op,const CartesianMoment& moment,
    bool mixed,std::span<Complex> centers,std::span<Complex> other) {
    switch (op) {
    case AtomOperator::overlap:
        if (mixed) {compute_overlap_mixed_derivatives(a,b,field,centers);}
        else {compute_overlap_center_derivatives(a,b,field,centers);}break;
    case AtomOperator::kinetic:
        if (mixed) {compute_kinetic_mixed_derivatives(a,b,field,centers);}
        else {compute_kinetic_center_derivatives(a,b,field,centers);}break;
    case AtomOperator::magnetic_kinetic:
        if (mixed) {compute_magnetic_kinetic_mixed_derivatives(a,b,field,centers);}
        else {compute_magnetic_kinetic_center_derivatives(a,b,field,centers);}break;
    case AtomOperator::attraction:
        if (mixed) {
            compute_nuclear_attraction_mixed_derivatives(a,b,nuclei,field,centers);
            compute_nuclear_attraction_nucleus_mixed_derivatives(a,b,nuclei,field,other);
        } else {
            compute_nuclear_attraction_center_derivatives(a,b,nuclei,field,centers);
            compute_nuclear_attraction_nucleus_derivatives(a,b,nuclei,field,other);
        }break;
    case AtomOperator::moment:
        if (mixed) {
            compute_moment_mixed_derivatives(a,b,moment,field,centers);
            compute_moment_origin_magnetic_derivatives(a,b,moment,field,other);
        } else {
            compute_moment_center_derivatives(a,b,moment,field,centers);
            compute_moment_origin_derivatives(a,b,moment,field,other);
        }break;
    }
}
} // namespace

void atom_one_electron_derivatives(const Basis& basis,std::span<const std::size_t> atoms,
    std::size_t natom,std::span<const Nucleus> nuclei,const MagneticField& field,
    AtomOperator op,const CartesianMoment& moment,bool mixed,std::span<Complex> output,
    std::span<Complex> origin,std::size_t threads) {
    validate_atoms(basis,atoms,natom);
    const auto n=basis.ao_count(),ns=basis.shells().size();
    const std::size_t components=mixed?9U:3U;
    const bool is_moment=op==AtomOperator::moment,is_attraction=op==AtomOperator::attraction;
    if (output.size()!=natom*components*n*n || origin.size()!=(is_moment?components*n*n:0U)) {
        throw std::length_error("atom derivative output has incorrect shape");
    }
    if (is_attraction && nuclei.size()!=natom) {throw std::invalid_argument("one attraction potential per atom is required");}
    std::fill(output.begin(),output.end(),Complex{});
    parallel_for(ns*ns,threads,[&](std::size_t pair) {
        const auto sa=pair/ns,sb=pair%ns;const auto& a=basis.shells()[sa];const auto& b=basis.shells()[sb];
        const auto size=shell_pair_size(a,b);
        const auto other_count=is_attraction?natom*components:(is_moment?components:0U);
        std::vector<Complex> centers(2U*components*size),other(other_count*size);
        pair_response(a,b,nuclei,field,op,moment,mixed,centers,other);
        for (std::size_t i=0;i<a.ao_count();++i) {
            for (std::size_t j=0;j<b.ao_count();++j) {
                const auto local=i*b.ao_count()+j;
                const auto global=(basis.ao_offsets()[sa]+i)*n+basis.ao_offsets()[sb]+j;
                for (std::size_t axis=0;axis<components;++axis) {
                    output[(atoms[sa]*components+axis)*n*n+global]+=centers[axis*size+local];
                    output[(atoms[sb]*components+axis)*n*n+global]+=centers[(components+axis)*size+local];
                    if (is_moment) {origin[axis*n*n+global]=other[axis*size+local];}
                }
                if (is_attraction) {
                    for (std::size_t atom=0;atom<natom;++atom) {
                        for (std::size_t axis=0;axis<components;++axis) {
                            output[(atom*components+axis)*n*n+global]+=other[(atom*components+axis)*size+local];
                        }
                    }
                }
            }
        }
    });
}

void contract_eri_derivatives(const Basis& basis,std::span<const std::size_t> atoms,
    std::size_t natom,std::span<const ShellQuartetIndex> quartets,
    std::span<const std::size_t> offsets,std::span<const Complex> values,
    std::span<const DensityPair> coulomb,std::span<const DensityPair> exchange,
    std::span<const Complex> gamma,std::span<const double> real_gamma,
    std::span<Complex> output,std::size_t threads) {
    validate_atoms(basis,atoms,natom);
    const auto n=basis.ao_count();
    if (output.size()!=natom*3U || offsets.size()!=quartets.size()+1U || offsets[0]!=0U || offsets.back()!=values.size()) {
        throw std::length_error("ERI derivative contraction buffer has incorrect shape");
    }
    for (const auto& pair:coulomb) {if(pair.p.size()!=n*n || pair.q.size()!=n*n){throw std::length_error("Coulomb density shape mismatch");}}
    for (const auto& pair:exchange) {if(pair.p.size()!=n*n || pair.q.size()!=n*n){throw std::length_error("exchange density shape mismatch");}}
    if ((!gamma.empty() && gamma.size()!=n*n*n*n) || (!real_gamma.empty() && real_gamma.size()!=n*n*n*n)) {
        throw std::length_error("two-particle density shape mismatch");
    }
    for (std::size_t row=0;row<quartets.size();++row) {
        const auto q=quartets[row];
        for (const auto s:q.as_array()) {if(s>=basis.shells().size()){throw std::out_of_range("quartet index out of range");}}
        if(canonicalize_shell_quartet(q).shells.as_array()!=q.as_array()){throw std::invalid_argument("contraction requires canonical quartets");}
        const auto size=shell_quartet_size(basis.shells()[q.a],basis.shells()[q.b],basis.shells()[q.c],basis.shells()[q.d]);
        if(offsets[row+1U]<offsets[row] || offsets[row+1U]-offsets[row]!=12U*size){throw std::length_error("derivative offsets mismatch");}
    }
    // Twelve partial sums per quartet, bounded by the packed derivative values.
    std::vector<std::array<Complex,12>> partials(quartets.size());
    constexpr std::array<std::array<std::size_t,4>,4> permutations{{{0,1,2,3},{2,3,0,1},{1,0,3,2},{3,2,1,0}}};
    parallel_for(quartets.size(),threads,[&](std::size_t row) {
        const auto q=quartets[row].as_array();
        std::array<std::size_t,4> shape{},starts{};
        std::size_t size=1U;
        for(std::size_t k=0;k<4U;++k){shape[k]=basis.shells()[q[k]].ao_count();starts[k]=basis.ao_offsets()[q[k]];size*=shape[k];}
        std::array<bool,4> active{true,true,true,true};
        for(std::size_t orbit=1;orbit<4U;++orbit){
            for(std::size_t previous=0;previous<orbit;++previous){
                bool same=true;for(std::size_t k=0;k<4U;++k){same=same && q[permutations[orbit][k]]==q[permutations[previous][k]];}
                if(same){active[orbit]=false;break;}
            }
        }
        const auto block=values.subspan(offsets[row],12U*size);
        for(std::size_t local=0;local<size;++local){
            std::size_t rest=local;std::array<std::size_t,4> ao{};
            for(std::size_t k=4U;k-->0U;){ao[k]=starts[k]+rest%shape[k];rest/=shape[k];}
            for(std::size_t orbit=0;orbit<4U;++orbit){
                if(!active[orbit]){continue;}
                const auto& perm=permutations[orbit];
                const auto mu=ao[perm[0]],nu=ao[perm[1]],lambda=ao[perm[2]],sigma=ao[perm[3]];
                Complex weight{};
                for(const auto& pair:coulomb){weight+=pair.p[nu*n+mu]*pair.q[sigma*n+lambda];}
                for(const auto& pair:exchange){weight+=pair.p[nu*n+lambda]*pair.q[sigma*n+mu];}
                const auto density_index=((mu*n+nu)*n+lambda)*n+sigma;
                if(!gamma.empty()){weight+=gamma[density_index];}
                if(!real_gamma.empty()){weight+=real_gamma[density_index];}
                for(std::size_t new_slot=0;new_slot<4U;++new_slot){
                    const auto slot=perm[new_slot];
                    for(std::size_t axis=0;axis<3U;++axis){
                        auto derivative=block[(slot*3U+axis)*size+local];
                        if(orbit>=2U){derivative=std::conj(derivative);}
                        partials[row][slot*3U+axis]+=derivative*weight;
                    }
                }
            }
        }
    });
    // Fixed canonical-quartet and slot order, independent of thread scheduling.
    for(std::size_t row=0;row<quartets.size();++row){
        const auto q=quartets[row].as_array();
        for(std::size_t slot=0;slot<4U;++slot){
            for(std::size_t axis=0;axis<3U;++axis){output[atoms[q[slot]]*3U+axis]+=partials[row][slot*3U+axis];}
        }
    }
}
} // namespace giao::detail

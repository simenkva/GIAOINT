#include "cholesky.hpp"
#include "eri_kernel.hpp"
#include "parallel.hpp"

#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace giao::detail {
namespace {

// Same value kernel, primitive loop order, normalization multiplication and
// contraction expression as compute_eri. Only unneeded AO entries are omitted.
class SelectedQuartet {
  public:
    SelectedQuartet(const Shell& a,const Shell& b,const Shell& c,const Shell& d,
                    const MagneticField& field)
        : shells_{&a,&b,&c,&d},field_(field) {
        for (std::size_t slot=0;slot<4U;++slot) {
            const auto& s=*shells_[slot];
            normalizations_[slot].resize(s.cartesian_count()*s.primitive_count());
            for (std::size_t component=0;component<s.cartesian_count();++component) {
                for (std::size_t primitive=0;primitive<s.primitive_count();++primitive) {
                    normalizations_[slot][component*s.primitive_count()+primitive]=
                        primitive_normalization(s.exponents()[primitive],s.components()[component]);
                }
            }
        }
    }
    Complex evaluate(const std::array<std::size_t,4>& aos,EriWorkspace& workspace) const {
        const auto& a=*shells_[0];const auto& b=*shells_[1];
        const auto& c=*shells_[2];const auto& d=*shells_[3];
        std::array<std::size_t,4> components{},contractions{};
        for (std::size_t slot=0;slot<4U;++slot) {
            components[slot]=aos[slot]%shells_[slot]->cartesian_count();
            contractions[slot]=aos[slot]/shells_[slot]->cartesian_count();
        }
        Complex result{};
        for (std::size_t pa=0;pa<a.primitive_count();++pa) {
            for (std::size_t pb=0;pb<b.primitive_count();++pb) {
                const auto pair_ab=gaussian_product(
                    PrimitiveGaussian(a.exponents()[pa],a.center(),{},1.0,false),
                    PrimitiveGaussian(b.exponents()[pb],b.center(),{},1.0,false),field_);
                for (std::size_t pc=0;pc<c.primitive_count();++pc) {
                    for (std::size_t pd=0;pd<d.primitive_count();++pd) {
                        const auto pair_cd=gaussian_product(
                            PrimitiveGaussian(c.exponents()[pc],c.center(),{},1.0,false),
                            PrimitiveGaussian(d.exponents()[pd],d.center(),{},1.0,false),field_);
                        const std::array<std::size_t,4> primitive_indices{pa,pb,pc,pd};
                        const std::array<PrimitiveGaussian,4> p{
                            PrimitiveGaussian(a.exponents()[pa],a.center(),a.components()[components[0]],1.0,false),
                            PrimitiveGaussian(b.exponents()[pb],b.center(),b.components()[components[1]],1.0,false),
                            PrimitiveGaussian(c.exponents()[pc],c.center(),c.components()[components[2]],1.0,false),
                            PrimitiveGaussian(d.exponents()[pd],d.center(),d.components()[components[3]],1.0,false)};
                        Complex primitive_value=EriKernel::compute(p[0],p[1],p[2],p[3],pair_ab,pair_cd,
                            EriKernel::use_real_path(field_),workspace);
                        for (std::size_t slot=0;slot<4U;++slot) {
                            primitive_value*=normalizations_[slot][components[slot]*shells_[slot]->primitive_count()+primitive_indices[slot]];
                        }
                        result+=a.coefficient(contractions[0],pa)*b.coefficient(contractions[1],pb)*
                                c.coefficient(contractions[2],pc)*d.coefficient(contractions[3],pd)*primitive_value;
                    }
                }
            }
        }
        return result;
    }
    // Contract all requested entries of this shell quartet primitive-first.
    // A component integral is independent of contraction rows: evaluate it
    // once, then distribute it without changing any AO's primitive sum order.
    std::vector<Complex> contracted_columns(
        std::span<const std::array<std::size_t,2>> local_pivots,
        EriWorkspace& workspace) const {
        const auto& a=*shells_[0]; const auto& b=*shells_[1];
        const auto& c=*shells_[2]; const auto& d=*shells_[3];
        const auto size=a.ao_count()*b.ao_count();
        std::vector<Complex> values(local_pivots.size()*size);
        std::array<std::vector<PrimitiveGaussian>,4> primitives;
        for (std::size_t slot=0;slot<4U;++slot) {
            const auto& shell=*shells_[slot];
            for (const auto exponent:shell.exponents()) {
                for (const auto angular:shell.components()) {
                    primitives[slot].emplace_back(exponent,shell.center(),angular,1.0,false);
                }
            }
        }
        // Bound pair caches even for unusually long general contractions.
        // Beyond this cap, compute each pair in the same primitive loop order.
        constexpr std::size_t maximum_cached_pairs=4096U;
        std::vector<GaussianPairData> pairs_ab,pairs_cd;
        if (a.primitive_count()<=maximum_cached_pairs/b.primitive_count()) {
            for (std::size_t pa=0;pa<a.primitive_count();++pa) {
                for (std::size_t pb=0;pb<b.primitive_count();++pb) {
                    pairs_ab.push_back(gaussian_product(primitives[0][pa*a.cartesian_count()],
                        primitives[1][pb*b.cartesian_count()],field_));
                }
            }
        }
        if (c.primitive_count()<=maximum_cached_pairs/d.primitive_count()) {
            for (std::size_t pc=0;pc<c.primitive_count();++pc) {
                for (std::size_t pd=0;pd<d.primitive_count();++pd) {
                    pairs_cd.push_back(gaussian_product(primitives[2][pc*c.cartesian_count()],
                        primitives[3][pd*d.cartesian_count()],field_));
                }
            }
        }
        std::vector<std::array<std::size_t,2>> components;
        std::vector<std::size_t> candidate_components;
        for (const auto& pivot:local_pivots) {
            const std::array<std::size_t,2> key{
                pivot[0]%c.cartesian_count(),pivot[1]%d.cartesian_count()};
            const auto found=std::find(components.begin(),components.end(),key);
            candidate_components.push_back(static_cast<std::size_t>(found-components.begin()));
            if (found==components.end()) {components.push_back(key);}
        }
        for (std::size_t pa=0;pa<a.primitive_count();++pa) {
            for (std::size_t pb=0;pb<b.primitive_count();++pb) {
                GaussianPairData temporary_ab{};
                const auto& ab=pairs_ab.empty()
                    ? (temporary_ab=gaussian_product(primitives[0][pa*a.cartesian_count()],
                        primitives[1][pb*b.cartesian_count()],field_))
                    : pairs_ab[pa*b.primitive_count()+pb];
                for (std::size_t pc=0;pc<c.primitive_count();++pc) {
                    for (std::size_t pd=0;pd<d.primitive_count();++pd) {
                        GaussianPairData temporary_cd{};
                        const auto& cd=pairs_cd.empty()
                            ? (temporary_cd=gaussian_product(primitives[2][pc*c.cartesian_count()],
                                primitives[3][pd*d.cartesian_count()],field_))
                            : pairs_cd[pc*d.primitive_count()+pd];
                        const std::array<std::size_t,4> ps{pa,pb,pc,pd};
                        for (std::size_t ca=0;ca<a.cartesian_count();++ca) {
                            for (std::size_t cb=0;cb<b.cartesian_count();++cb) {
                                for (std::size_t k=0;k<components.size();++k) {
                                    const auto cc=components[k][0],dc=components[k][1];
                                    Complex value=EriKernel::compute(
                                        primitives[0][pa*a.cartesian_count()+ca],
                                        primitives[1][pb*b.cartesian_count()+cb],
                                        primitives[2][pc*c.cartesian_count()+cc],
                                        primitives[3][pd*d.cartesian_count()+dc],
                                        ab,cd,EriKernel::use_real_path(field_),workspace);
                                    const std::array<std::size_t,4> cs{ca,cb,cc,dc};
                                    for (std::size_t slot=0;slot<4U;++slot) {
                                        value*=normalizations_[slot][cs[slot]*shells_[slot]->primitive_count()+ps[slot]];
                                    }
                                    for (std::size_t candidate=0;candidate<local_pivots.size();++candidate) {
                                        if (candidate_components[candidate]!=k) {continue;}
                                        const auto xc=local_pivots[candidate][0]/c.cartesian_count();
                                        const auto xd=local_pivots[candidate][1]/d.cartesian_count();
                                        for (std::size_t xa=0;xa<a.contraction_count();++xa) {
                                            const auto ia=xa*a.cartesian_count()+ca;
                                            for (std::size_t xb=0;xb<b.contraction_count();++xb) {
                                                const auto ib=xb*b.cartesian_count()+cb;
                                                values[candidate*size+ia*b.ao_count()+ib]+=
                                                    a.coefficient(xa,pa)*b.coefficient(xb,pb)*
                                                    c.coefficient(xc,pc)*d.coefficient(xd,pd)*value;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return values;
    }
  private:
    std::array<const Shell*,4> shells_;
    MagneticField field_;
    std::array<std::vector<double>,4> normalizations_;
};

bool needs_contraction(const Shell& shell) noexcept {
    return shell.contraction_count()>1U || shell.primitive_count()>1U;
}

struct Task {std::size_t group;std::size_t begin;std::size_t end;};
struct Group {
    std::size_t a,b;
    std::unique_ptr<SelectedQuartet> quartet;
    bool contracted{};
};

void append_tasks(std::vector<Task>& tasks,std::size_t group,std::size_t count) {
    constexpr std::size_t chunk=64U;
    for (std::size_t begin=0;begin<count;begin+=chunk) {
        tasks.push_back({group,begin,std::min(count,begin+chunk)});
    }
}

std::vector<std::size_t> shell_owners(const Basis& basis) {
    std::vector<std::size_t> result(basis.ao_count());
    for (std::size_t s=0;s<basis.shells().size();++s) {
        std::fill_n(result.begin()+static_cast<std::ptrdiff_t>(basis.ao_offsets()[s]),basis.shells()[s].ao_count(),s);
    }
    return result;
}

template <typename Value>
void columns(const Basis& basis,std::span<const std::array<std::size_t,2>> pivots,
             const MagneticField& field,std::span<Value> output,std::size_t threads) {
    const auto n=basis.ao_count();
    if (output.size()!=pivots.size()*n*n) {throw std::length_error("selected Cholesky column output has incorrect size");}
    const bool real=EriKernel::use_real_path(field);
    if constexpr (std::is_same_v<Value,double>) {
        if (!real) {throw std::invalid_argument("real Cholesky columns require zero phase field");}
    }
    if (pivots.empty()) {parallel_for(0U,threads,[](std::size_t){});return;}
    const auto owners=shell_owners(basis);
    for (const auto& p:pivots) {
        if (p[0]>=n || p[1]>=n) {throw std::out_of_range("Cholesky pivot AO index is out of range");}
    }
    const auto sa=owners[pivots[0][0]],sb=owners[pivots[0][1]];
    for (const auto& p:pivots) {
        if (owners[p[0]]!=sa || owners[p[1]]!=sb) {throw std::invalid_argument("Cholesky candidate pivots must share a shell pair");}
    }
    std::vector<Group> groups;
    std::vector<Task> tasks;
    for (std::size_t a=0;a<basis.shells().size();++a) {
        for (std::size_t b=real?a:0U;b<basis.shells().size();++b) {
            const auto& left=basis.shells()[a];const auto& right=basis.shells()[b];
            groups.push_back({a,b,std::make_unique<SelectedQuartet>(left,right,basis.shells()[sb],basis.shells()[sa],field)});
            const bool contracted=needs_contraction(left) || needs_contraction(right) ||
                needs_contraction(basis.shells()[sa]) || needs_contraction(basis.shells()[sb]);
            groups.back().contracted=contracted;
            if (contracted) {tasks.push_back({groups.size()-1U,0U,1U});}
            else {append_tasks(tasks,groups.size()-1U,pivots.size()*left.ao_count()*right.ao_count());}
        }
    }
    parallel_for(tasks.size(),threads,[&](std::size_t index) {
        const auto task=tasks[index];const auto& group=groups[task.group];
        const auto& a=basis.shells()[group.a];const auto& b=basis.shells()[group.b];
        const auto pair_size=a.ao_count()*b.ao_count();
        EriWorkspace workspace;
        if (group.contracted) {
            std::vector<std::array<std::size_t,2>> local;
            for (const auto& p:pivots) {
                local.push_back({p[1]-basis.ao_offsets()[sb],p[0]-basis.ao_offsets()[sa]});
            }
            const auto values=group.quartet->contracted_columns(local,workspace);
            for (std::size_t candidate=0;candidate<pivots.size();++candidate) {
                for (std::size_t i=0;i<a.ao_count();++i) {
                    for (std::size_t j=0;j<b.ao_count();++j) {
                        const auto value=values[candidate*pair_size+i*b.ao_count()+j];
                        Value stored;
                        if constexpr (std::is_same_v<Value,double>) {
                            stored=value.real();
                            if (group.a==group.b) {
                                stored=(stored+values[candidate*pair_size+j*b.ao_count()+i].real())*0.5;
                            }
                        } else {stored=value;}
                        const auto mu=basis.ao_offsets()[group.a]+i,nu=basis.ao_offsets()[group.b]+j;
                        output[(candidate*n+mu)*n+nu]=stored;
                        if (real && group.a!=group.b) {output[(candidate*n+nu)*n+mu]=stored;}
                    }
                }
            }
            return;
        }
        for (std::size_t point=task.begin;point<task.end;++point) {
            const auto candidate=point/pair_size;
            const auto i=(point%pair_size)/b.ao_count(),j=point%b.ao_count();
            const auto c=pivots[candidate][1]-basis.ao_offsets()[sb];
            const auto d=pivots[candidate][0]-basis.ao_offsets()[sa];
            const auto value=group.quartet->evaluate({i,j,c,d},workspace);
            Value stored;
            if constexpr (std::is_same_v<Value,double>) {
                stored=value.real();
                if (group.a==group.b) {
                    stored=(stored+group.quartet->evaluate({j,i,c,d},workspace).real())*0.5;
                }
            } else {stored=value;}
            const auto mu=basis.ao_offsets()[group.a]+i,nu=basis.ao_offsets()[group.b]+j;
            output[(candidate*n+mu)*n+nu]=stored;
            if (real && group.a!=group.b) {output[(candidate*n+nu)*n+mu]=stored;}
        }
    });
}
} // namespace

void cholesky_diagonal(const Basis& basis,std::span<const std::array<std::size_t,2>> pairs,
                      const MagneticField& field,std::span<double> output,std::size_t threads) {
    const auto n=basis.ao_count(),ns=basis.shells().size();
    if (output.size()!=n*n) {throw std::length_error("Cholesky diagonal output has incorrect size");}
    std::fill(output.begin(),output.end(),0.0);
    std::vector<bool> seen(ns*ns);
    std::vector<Group> groups;
    std::vector<Task> tasks;
    for (const auto& pair:pairs) {
        const auto a=pair[0],b=pair[1];
        if (a>=ns || b>=ns || a>b) {throw std::out_of_range("Cholesky diagonal shell pair must be valid and ordered");}
        if (seen[a*ns+b]) {throw std::invalid_argument("duplicate Cholesky diagonal shell pair");}
        seen[a*ns+b]=true;
        groups.push_back({a,b,std::make_unique<SelectedQuartet>(basis.shells()[a],basis.shells()[b],basis.shells()[b],basis.shells()[a],field)});
        append_tasks(tasks,groups.size()-1U,basis.shells()[a].ao_count()*basis.shells()[b].ao_count());
    }
    parallel_for(tasks.size(),threads,[&](std::size_t index) {
        const auto task=tasks[index];const auto& group=groups[task.group];
        const auto& b=basis.shells()[group.b];
        EriWorkspace workspace;
        for (std::size_t point=task.begin;point<task.end;++point) {
            const auto i=point/b.ao_count(),j=point%b.ao_count();
            const double value=group.quartet->evaluate({i,j,j,i},workspace).real();
            const auto mu=basis.ao_offsets()[group.a]+i,nu=basis.ao_offsets()[group.b]+j;
            // read_diagonal historically transposes same-shell blocks last.
            output[nu*n+mu]=value;
            if (group.a!=group.b) {output[mu*n+nu]=value;}
        }
    });
}

void cholesky_columns(const Basis& basis,std::span<const std::array<std::size_t,2>> pivots,
                     const MagneticField& field,std::span<double> output,std::size_t threads) {
    columns(basis,pivots,field,output,threads);
}
void cholesky_columns(const Basis& basis,std::span<const std::array<std::size_t,2>> pivots,
                     const MagneticField& field,std::span<Complex> output,std::size_t threads) {
    columns(basis,pivots,field,output,threads);
}
} // namespace giao::detail

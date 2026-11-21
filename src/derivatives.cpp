#include "giao_integrals/derivatives.hpp"
#include "parallel.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

namespace giao {
namespace {

constexpr std::size_t axis_count = 3U;

CartesianExponent shifted(CartesianExponent angular, std::size_t axis, int amount) {
    auto values = std::array<std::uint16_t, 3>{angular.x, angular.y, angular.z};
    const int value = static_cast<int>(values[axis]) + amount;
    if (value < 0 || value > std::numeric_limits<std::uint16_t>::max()) {
        throw std::overflow_error("shifted angular momentum is out of range");
    }
    values[axis] = static_cast<std::uint16_t>(value);
    return {values[0], values[1], values[2]};
}

PrimitiveGaussian raw_primitive(const PrimitiveGaussian& primitive) {
    const double normalization =
        primitive.normalized
            ? primitive_normalization(primitive.exponent, primitive.angular)
            : 1.0;
    return PrimitiveGaussian(primitive.exponent, primitive.center, primitive.angular,
                             primitive.coefficient * normalization, false);
}

PrimitiveGaussian with_angular(const PrimitiveGaussian& raw,
                               CartesianExponent angular) {
    return PrimitiveGaussian(raw.exponent, raw.center, angular, raw.coefficient, false);
}

Vec3 cross(Vec3 left, Vec3 right) noexcept {
    return {left.y * right.z - left.z * right.y, left.z * right.x - left.x * right.z,
            left.x * right.y - left.y * right.x};
}

Vec3 unit_vector(std::size_t axis) noexcept {
    return {axis == 0U ? 1.0 : 0.0, axis == 1U ? 1.0 : 0.0, axis == 2U ? 1.0 : 0.0};
}

template <typename Evaluator>
Complex center_response(const PrimitiveGaussian& raw, bool bra,
                        std::size_t derivative_axis, const MagneticField& field,
                        Evaluator&& evaluate) {
    Complex result{};
    const auto angular = raw.angular[derivative_axis];
    if (angular != 0U) {
        result -=
            static_cast<double>(angular) *
            evaluate(with_angular(raw, shifted(raw.angular, derivative_axis, -1)));
    }
    result += 2.0 * raw.exponent *
              evaluate(with_angular(raw, shifted(raw.angular, derivative_axis, 1)));

    const Vec3 phase_derivative = cross(field.B, unit_vector(derivative_axis));
    const double phase_sign = bra ? 0.5 : -0.5;
    const Complex base = evaluate(raw);
    for (std::size_t coordinate = 0; coordinate < axis_count; ++coordinate) {
        const double coefficient = phase_sign * phase_derivative[coordinate];
        if (coefficient == 0.0) {
            continue;
        }
        const Complex coordinate_value =
            evaluate(with_angular(raw, shifted(raw.angular, coordinate, 1))) +
            raw.center[coordinate] * base;
        result += Complex{0.0, coefficient} * coordinate_value;
    }
    return result;
}

template <typename Evaluator>
Complex magnetic_response(const PrimitiveGaussian& raw, bool bra,
                          std::size_t field_axis, const MagneticField& field,
                          Evaluator&& evaluate) {
    const Vec3 displacement{raw.center.x - field.gauge_origin.x,
                            raw.center.y - field.gauge_origin.y,
                            raw.center.z - field.gauge_origin.z};
    const Vec3 phase_derivative = cross(unit_vector(field_axis), displacement);
    const double phase_sign = bra ? 0.5 : -0.5;
    const Complex base = evaluate(raw);
    Complex result{};
    for (std::size_t coordinate = 0; coordinate < axis_count; ++coordinate) {
        const double coefficient = phase_sign * phase_derivative[coordinate];
        if (coefficient == 0.0) {
            continue;
        }
        const Complex coordinate_value =
            evaluate(with_angular(raw, shifted(raw.angular, coordinate, 1))) +
            raw.center[coordinate] * base;
        result += Complex{0.0, coefficient} * coordinate_value;
    }
    return result;
}

template <typename Evaluator>
Complex spatial_response(const PrimitiveGaussian& raw, bool bra, std::size_t axis,
                         const MagneticField& field, Evaluator&& evaluate) {
    Complex result{};
    const auto angular = raw.angular[axis];
    if (angular != 0U) {
        result += static_cast<double>(angular) *
                  evaluate(with_angular(raw, shifted(raw.angular, axis, -1)));
    }
    result -=
        2.0 * raw.exponent * evaluate(with_angular(raw, shifted(raw.angular, axis, 1)));
    const double phase_sign = bra ? 1.0 : -1.0;
    result += Complex{0.0, phase_sign * field.london_wave_vector(raw.center)[axis]} *
              evaluate(raw);
    return result;
}

template <typename Evaluator>
std::array<Complex, 6>
pair_center_derivatives(const PrimitiveGaussian& bra, const PrimitiveGaussian& ket,
                        const MagneticField& field, Evaluator&& evaluate) {
    const auto raw_bra = raw_primitive(bra);
    const auto raw_ket = raw_primitive(ket);
    std::array<Complex, 6> result{};
    for (std::size_t axis = 0; axis < axis_count; ++axis) {
        result[axis] =
            center_response(raw_bra, true, axis, field, [&](const auto& changed) {
                return evaluate(changed, raw_ket);
            });
        result[axis_count + axis] =
            center_response(raw_ket, false, axis, field, [&](const auto& changed) {
                return evaluate(raw_bra, changed);
            });
    }
    return result;
}

template <typename Evaluator>
std::array<Complex, 3>
pair_magnetic_derivatives(const PrimitiveGaussian& bra, const PrimitiveGaussian& ket,
                          const MagneticField& field, Evaluator&& evaluate) {
    const auto raw_bra = raw_primitive(bra);
    const auto raw_ket = raw_primitive(ket);
    std::array<Complex, 3> result{};
    for (std::size_t axis = 0; axis < axis_count; ++axis) {
        result[axis] =
            magnetic_response(
                raw_bra, true, axis, field,
                [&](const auto& changed) { return evaluate(changed, raw_ket); }) +
            magnetic_response(raw_ket, false, axis, field, [&](const auto& changed) {
                return evaluate(raw_bra, changed);
            });
    }
    return result;
}


template <typename Evaluator>
Complex center_phase_field_response(const PrimitiveGaussian& raw, bool bra,
    std::size_t nuclear_axis, std::size_t field_axis, Evaluator&& evaluate) {
    const auto coefficient = cross(unit_vector(field_axis), unit_vector(nuclear_axis));
    const double sign = bra ? 0.5 : -0.5;
    Complex result{};
    const Complex base = evaluate(raw);
    for (std::size_t coordinate = 0; coordinate < 3U; ++coordinate) {
        if (coefficient[coordinate] != 0.0) {
            result += Complex{0.0, sign * coefficient[coordinate]} *
                (evaluate(with_angular(raw, shifted(raw.angular, coordinate, 1))) +
                 raw.center[coordinate] * base);
        }
    }
    return result;
}

template <typename Value, typename Magnetic>
std::array<Complex, 18> pair_mixed_derivatives(const PrimitiveGaussian& bra,
    const PrimitiveGaussian& ket, const MagneticField& field,
    Value&& value, Magnetic&& magnetic) {
    const std::array<PrimitiveGaussian,2> raw{raw_primitive(bra),raw_primitive(ket)};
    // A shifted pair is reused across field/nuclear axes. Cache its complete
    // field response, preserving the value kernels and summation order.
    std::map<std::array<std::uint16_t,6>,std::array<Complex,3>> magnetic_cache;
    auto cached_magnetic = [&](const auto& a,const auto& b) -> const std::array<Complex,3>& {
        const std::array<std::uint16_t,6> key{a.angular.x,a.angular.y,a.angular.z,
                                              b.angular.x,b.angular.y,b.angular.z};
        auto [it, inserted]=magnetic_cache.try_emplace(key);
        if (inserted) { it->second=magnetic(a,b); }
        return it->second;
    };
    std::array<Complex,18> result{};
    for (std::size_t center = 0; center < 2U; ++center) {
        for (std::size_t r = 0; r < 3U; ++r) {
            for (std::size_t b = 0; b < 3U; ++b) {
                auto eval_b = [&](const auto& changed) {
                    return center == 0U ? cached_magnetic(changed,raw[1])[b] : cached_magnetic(raw[0],changed)[b];
                };
                auto eval = [&](const auto& changed) {
                    return center == 0U ? value(changed,raw[1]) : value(raw[0],changed);
                };
                result[(center*3U+r)*3U+b] =
                    center_response(raw[center],center==0U,r,field,eval_b) +
                    center_phase_field_response(raw[center],center==0U,r,b,eval);
            }
        }
    }
    return result;
}
Complex coordinate_momentum(const PrimitiveGaussian& raw_bra,
                            const PrimitiveGaussian& raw_ket,
                            std::size_t coordinate_axis, std::size_t momentum_axis,
                            const MagneticField& field, IntegralWorkspace& workspace) {
    const auto raised =
        with_angular(raw_bra, shifted(raw_bra.angular, coordinate_axis, 1));
    return primitive_momentum(raised, raw_ket, static_cast<Axis>(momentum_axis), field,
                              workspace) +
           (raw_bra.center[coordinate_axis] - field.gauge_origin[coordinate_axis]) *
               primitive_momentum(raw_bra, raw_ket, static_cast<Axis>(momentum_axis),
                                  field, workspace);
}

Complex explicit_magnetic_kinetic_response(const PrimitiveGaussian& raw_bra,
                                           const PrimitiveGaussian& raw_ket,
                                           std::size_t field_axis,
                                           const MagneticField& field,
                                           IntegralWorkspace& workspace) {
    const std::size_t first = (field_axis + 1U) % axis_count;
    const std::size_t second = (field_axis + 2U) % axis_count;
    Complex result =
        0.5 * (coordinate_momentum(raw_bra, raw_ket, first, second, field, workspace) -
               coordinate_momentum(raw_bra, raw_ket, second, first, field, workspace));

    Complex radius_squared{};
    Complex projected_coordinate{};
    for (std::size_t axis = 0; axis < axis_count; ++axis) {
        auto square = CartesianExponent{};
        if (axis == 0U) {
            square.x = 2U;
        } else if (axis == 1U) {
            square.y = 2U;
        } else {
            square.z = 2U;
        }
        radius_squared += primitive_moment(raw_bra, raw_ket,
                                           CartesianMoment(field.gauge_origin, square),
                                           field, workspace);

        auto product = CartesianExponent{};
        if (axis == field_axis) {
            if (axis == 0U) {
                product.x = 2U;
            } else if (axis == 1U) {
                product.y = 2U;
            } else {
                product.z = 2U;
            }
        } else {
            if (axis == 0U || field_axis == 0U) {
                product.x = 1U;
            }
            if (axis == 1U || field_axis == 1U) {
                product.y = 1U;
            }
            if (axis == 2U || field_axis == 2U) {
                product.z = 1U;
            }
        }
        projected_coordinate +=
            field.B[axis] *
            primitive_moment(raw_bra, raw_ket,
                             CartesianMoment(field.gauge_origin, product), field,
                             workspace);
    }
    result += 0.25 * (field.B[field_axis] * radius_squared - projected_coordinate);
    return result;
}

template <std::size_t derivative_count, typename Evaluator>
void compute_pair_block(const Shell& a, const Shell& b, std::span<Complex> output,
                        Evaluator&& evaluate) {
    const std::size_t block_size = shell_pair_size(a, b);
    if (output.size() != derivative_count * block_size) {
        throw std::length_error("derivative shell-pair output has an incorrect size");
    }
    std::fill(output.begin(), output.end(), Complex{});
    const auto components_a = a.cartesian_count();
    const auto components_b = b.cartesian_count();
    for (std::size_t primitive_a = 0; primitive_a < a.primitive_count();
         ++primitive_a) {
        for (std::size_t primitive_b = 0; primitive_b < b.primitive_count();
             ++primitive_b) {
            for (std::size_t component_a = 0; component_a < components_a;
                 ++component_a) {
                const PrimitiveGaussian bra(
                    a.exponents()[primitive_a], a.center(), a.components()[component_a],
                    primitive_normalization(a.exponents()[primitive_a],
                                            a.components()[component_a]),
                    false);
                for (std::size_t component_b = 0; component_b < components_b;
                     ++component_b) {
                    const PrimitiveGaussian ket(
                        b.exponents()[primitive_b], b.center(),
                        b.components()[component_b],
                        primitive_normalization(b.exponents()[primitive_b],
                                                b.components()[component_b]),
                        false);
                    const auto values = evaluate(bra, ket);
                    for (std::size_t contraction_a = 0;
                         contraction_a < a.contraction_count(); ++contraction_a) {
                        const std::size_t ao_a =
                            contraction_a * components_a + component_a;
                        const double coefficient_a =
                            a.coefficient(contraction_a, primitive_a);
                        for (std::size_t contraction_b = 0;
                             contraction_b < b.contraction_count(); ++contraction_b) {
                            const std::size_t ao_b =
                                contraction_b * components_b + component_b;
                            const double weight =
                                coefficient_a *
                                b.coefficient(contraction_b, primitive_b);
                            const std::size_t element = ao_a * b.ao_count() + ao_b;
                            for (std::size_t derivative = 0;
                                 derivative < derivative_count; ++derivative) {
                                output[derivative * block_size + element] +=
                                    weight * values[derivative];
                            }
                        }
                    }
                }
            }
        }
    }
}

template <typename ShellEvaluator>
void compute_basis_nuclear_derivatives(const Basis& basis, std::span<Complex> output,
                                       ShellEvaluator&& evaluate) {
    const std::size_t count = basis.ao_count();
    const std::size_t shell_count = basis.shells().size();
    if (output.size() != shell_count * axis_count * count * count) {
        throw std::length_error(
            "nuclear-derivative basis output has an incorrect size");
    }
    std::fill(output.begin(), output.end(), Complex{});
    std::vector<Complex> block;
    for (std::size_t shell_a = 0; shell_a < shell_count; ++shell_a) {
        const auto& a = basis.shells()[shell_a];
        for (std::size_t shell_b = 0; shell_b < shell_count; ++shell_b) {
            const auto& b = basis.shells()[shell_b];
            const std::size_t pair_size = shell_pair_size(a, b);
            block.resize(2U * axis_count * pair_size);
            evaluate(a, b, block);
            for (std::size_t center = 0; center < 2U; ++center) {
                const std::size_t target_shell = center == 0U ? shell_a : shell_b;
                for (std::size_t axis = 0; axis < axis_count; ++axis) {
                    for (std::size_t ao_a = 0; ao_a < a.ao_count(); ++ao_a) {
                        for (std::size_t ao_b = 0; ao_b < b.ao_count(); ++ao_b) {
                            const std::size_t global_a =
                                basis.ao_offsets()[shell_a] + ao_a;
                            const std::size_t global_b =
                                basis.ao_offsets()[shell_b] + ao_b;
                            const std::size_t target =
                                ((target_shell * axis_count + axis) * count +
                                 global_a) *
                                    count +
                                global_b;
                            const std::size_t source =
                                (center * axis_count + axis) * pair_size +
                                ao_a * b.ao_count() + ao_b;
                            output[target] += block[source];
                        }
                    }
                }
            }
        }
    }
}

template <typename ShellEvaluator>
void compute_basis_magnetic_derivatives(const Basis& basis, std::span<Complex> output,
                                        ShellEvaluator&& evaluate) {
    const std::size_t count = basis.ao_count();
    if (output.size() != axis_count * count * count) {
        throw std::length_error(
            "magnetic-derivative basis output has an incorrect size");
    }
    std::fill(output.begin(), output.end(), Complex{});
    std::vector<Complex> block;
    for (std::size_t shell_a = 0; shell_a < basis.shells().size(); ++shell_a) {
        const auto& a = basis.shells()[shell_a];
        for (std::size_t shell_b = 0; shell_b < basis.shells().size(); ++shell_b) {
            const auto& b = basis.shells()[shell_b];
            const std::size_t pair_size = shell_pair_size(a, b);
            block.resize(axis_count * pair_size);
            evaluate(a, b, block);
            for (std::size_t axis = 0; axis < axis_count; ++axis) {
                for (std::size_t ao_a = 0; ao_a < a.ao_count(); ++ao_a) {
                    for (std::size_t ao_b = 0; ao_b < b.ao_count(); ++ao_b) {
                        const std::size_t global_a = basis.ao_offsets()[shell_a] + ao_a;
                        const std::size_t global_b = basis.ao_offsets()[shell_b] + ao_b;
                        output[(axis * count + global_a) * count + global_b] =
                            block[axis * pair_size + ao_a * b.ao_count() + ao_b];
                    }
                }
            }
        }
    }
}

// Assemble arbitrary derivative components without AO symmetry assumptions.
// Every shell pair owns disjoint AO entries, including both coincident centres.
template <typename Evaluator>
void property_derivative_matrices(const Basis& basis, std::size_t other_count,
    std::span<Complex> centers, std::span<Complex> other, std::size_t threads,
    Evaluator&& evaluate, std::size_t components = 3U) {
    const auto nshell = basis.shells().size();
    const auto nao = basis.ao_count();
    if (centers.size() != nshell * components * nao * nao ||
        other.size() != other_count * nao * nao) {
        throw std::length_error("property derivative matrix output has an incorrect size");
    }
    std::fill(centers.begin(), centers.end(), Complex{});
    std::fill(other.begin(), other.end(), Complex{});
    detail::parallel_for(nshell * nshell, threads, [&](std::size_t pair) {
        const auto sa = pair / nshell;
        const auto sb = pair % nshell;
        const auto& a = basis.shells()[sa];
        const auto& b = basis.shells()[sb];
        const auto size = shell_pair_size(a, b);
        std::vector<Complex> block((2U * components + other_count) * size);
        evaluate(a, b, block);
        for (std::size_t i = 0; i < a.ao_count(); ++i) {
            for (std::size_t j = 0; j < b.ao_count(); ++j) {
                const auto local = i * b.ao_count() + j;
                const auto global = (basis.ao_offsets()[sa] + i) * nao +
                                    basis.ao_offsets()[sb] + j;
                for (std::size_t axis = 0; axis < components; ++axis) {
                    centers[(sa * components + axis) * nao * nao + global] += block[axis * size + local];
                    centers[(sb * components + axis) * nao * nao + global] += block[(components + axis) * size + local];
                }
                for (std::size_t k = 0; k < other_count; ++k) {
                    other[k * nao * nao + global] = block[(2U * components + k) * size + local];
                }
            }
        }
    });
}

template <typename Evaluator>
void property_magnetic_matrices(const Basis& basis, std::span<Complex> output,
    std::size_t threads, Evaluator&& evaluate) {
    const auto nshell = basis.shells().size();
    const auto nao = basis.ao_count();
    if (output.size() != 3U * nao * nao) {
        throw std::length_error("property magnetic matrix output has an incorrect size");
    }
    detail::parallel_for(nshell * nshell, threads, [&](std::size_t pair) {
        const auto sa = pair / nshell;
        const auto sb = pair % nshell;
        const auto& a = basis.shells()[sa];
        const auto& b = basis.shells()[sb];
        const auto size = shell_pair_size(a, b);
        std::vector<Complex> block(3U * size);
        evaluate(a, b, block);
        for (std::size_t axis = 0; axis < 3U; ++axis) {
            for (std::size_t i = 0; i < a.ao_count(); ++i) {
                for (std::size_t j = 0; j < b.ao_count(); ++j) {
                    output[(axis * nao + basis.ao_offsets()[sa] + i) * nao +
                           basis.ao_offsets()[sb] + j] = block[axis * size + i * b.ao_count() + j];
                }
            }
        }
    });
}
template <std::size_t derivative_count, typename Evaluator>
void compute_quartet_block(const Shell& a, const Shell& b, const Shell& c,
    const Shell& d, std::span<Complex> output, Evaluator&& evaluate) {
    const std::size_t block_size = shell_quartet_size(a, b, c, d);
    if (output.size() != derivative_count * block_size) {
        throw std::length_error(
            "center-derivative shell-quartet output has an incorrect size");
    }
    std::fill(output.begin(), output.end(), Complex{});
    const std::array<const Shell*, 4> shells{&a, &b, &c, &d};
    std::array<std::vector<double>, 4> normalizations;
    for (std::size_t shell = 0; shell < 4U; ++shell) {
        normalizations[shell].resize(shells[shell]->cartesian_count() *
                                     shells[shell]->primitive_count());
        for (std::size_t component = 0; component < shells[shell]->cartesian_count();
             ++component) {
            for (std::size_t primitive = 0;
                 primitive < shells[shell]->primitive_count(); ++primitive) {
                normalizations[shell][component * shells[shell]->primitive_count() +
                                      primitive] =
                    primitive_normalization(shells[shell]->exponents()[primitive],
                                            shells[shell]->components()[component]);
            }
        }
    }
    for (std::size_t pa = 0; pa < a.primitive_count(); ++pa) {
        for (std::size_t pb = 0; pb < b.primitive_count(); ++pb) {
            for (std::size_t pc = 0; pc < c.primitive_count(); ++pc) {
                for (std::size_t pd = 0; pd < d.primitive_count(); ++pd) {
                    for (std::size_t ca = 0; ca < a.cartesian_count(); ++ca) {
                        for (std::size_t cb = 0; cb < b.cartesian_count(); ++cb) {
                            for (std::size_t cc = 0; cc < c.cartesian_count(); ++cc) {
                                for (std::size_t cd = 0; cd < d.cartesian_count();
                                     ++cd) {
                                    const PrimitiveGaussian ga(
                                        a.exponents()[pa], a.center(),
                                        a.components()[ca],
                                        normalizations[0]
                                                      [ca * a.primitive_count() + pa],
                                        false);
                                    const PrimitiveGaussian gb(
                                        b.exponents()[pb], b.center(),
                                        b.components()[cb],
                                        normalizations[1]
                                                      [cb * b.primitive_count() + pb],
                                        false);
                                    const PrimitiveGaussian gc(
                                        c.exponents()[pc], c.center(),
                                        c.components()[cc],
                                        normalizations[2]
                                                      [cc * c.primitive_count() + pc],
                                        false);
                                    const PrimitiveGaussian gd(
                                        d.exponents()[pd], d.center(),
                                        d.components()[cd],
                                        normalizations[3]
                                                      [cd * d.primitive_count() + pd],
                                        false);
                                    const auto values =
                                        evaluate(ga, gb, gc, gd);
                                    for (std::size_t xa = 0; xa < a.contraction_count();
                                         ++xa) {
                                        const auto aoa = xa * a.cartesian_count() + ca;
                                        for (std::size_t xb = 0;
                                             xb < b.contraction_count(); ++xb) {
                                            const auto aob =
                                                xb * b.cartesian_count() + cb;
                                            for (std::size_t xc = 0;
                                                 xc < c.contraction_count(); ++xc) {
                                                const auto aoc =
                                                    xc * c.cartesian_count() + cc;
                                                for (std::size_t xd = 0;
                                                     xd < d.contraction_count(); ++xd) {
                                                    const auto aod =
                                                        xd * d.cartesian_count() + cd;
                                                    const double weight =
                                                        a.coefficient(xa, pa) *
                                                        b.coefficient(xb, pb) *
                                                        c.coefficient(xc, pc) *
                                                        d.coefficient(xd, pd);
                                                    const std::size_t element =
                                                        ((aoa * b.ao_count() + aob) *
                                                             c.ao_count() +
                                                         aoc) *
                                                            d.ao_count() +
                                                        aod;
                                                    for (std::size_t derivative = 0;
                                                         derivative < derivative_count;
                                                         ++derivative) {
                                                        output[derivative * block_size +
                                                               element] +=
                                                            weight * values[derivative];
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
            }
        }
    }
}


} // namespace

std::array<Complex, 6> primitive_moment_center_derivatives(
    const PrimitiveGaussian& bra, const PrimitiveGaussian& ket,
    const CartesianMoment& moment, const MagneticField& field) {
    IntegralWorkspace workspace;
    return pair_center_derivatives(bra, ket, field, [&](const auto& a, const auto& b) {
        return primitive_moment(a, b, moment, field, workspace);
    });
}

std::array<Complex, 3> primitive_moment_origin_derivatives(
    const PrimitiveGaussian& bra, const PrimitiveGaussian& ket,
    const CartesianMoment& moment, const MagneticField& field) {
    IntegralWorkspace workspace;
    std::array<Complex, 3> result{};
    for (std::size_t axis = 0; axis < 3U; ++axis) {
        if (moment.powers[axis] != 0U) {
            result[axis] = -static_cast<double>(moment.powers[axis]) *
                primitive_moment(bra, ket, CartesianMoment(moment.origin,
                    shifted(moment.powers, axis, -1)), field, workspace);
        }
    }
    return result;
}

void compute_moment_center_derivatives(const Shell& a, const Shell& b,
    const CartesianMoment& moment, const MagneticField& field, std::span<Complex> output) {
    compute_pair_block<6U>(a, b, output, [&](const auto& bra, const auto& ket) {
        return primitive_moment_center_derivatives(bra, ket, moment, field);
    });
}

void compute_moment_origin_derivatives(const Shell& a, const Shell& b,
    const CartesianMoment& moment, const MagneticField& field, std::span<Complex> output) {
    compute_pair_block<3U>(a, b, output, [&](const auto& bra, const auto& ket) {
        return primitive_moment_origin_derivatives(bra, ket, moment, field);
    });
}

void compute_moment_nuclear_derivative_matrices(const Basis& basis,
    const CartesianMoment& moment, const MagneticField& field,
    std::span<Complex> centers, std::span<Complex> origin, std::size_t threads) {
    property_derivative_matrices(basis, 3U, centers, origin, threads,
        [&](const Shell& a, const Shell& b, std::span<Complex> output) {
            const auto size = shell_pair_size(a, b);
            compute_moment_center_derivatives(a, b, moment, field, output.first(6U * size));
            compute_moment_origin_derivatives(a, b, moment, field, output.subspan(6U * size));
        });
}

std::array<Complex, 6>
primitive_overlap_center_derivatives(const PrimitiveGaussian& bra,
                                     const PrimitiveGaussian& ket,
                                     const MagneticField& field) {
    IntegralWorkspace workspace;
    return pair_center_derivatives(
        bra, ket, field, [&](const auto& left, const auto& right) {
            return primitive_overlap(left, right, field, workspace);
        });
}

std::array<Complex, 3>
primitive_overlap_magnetic_derivatives(const PrimitiveGaussian& bra,
                                       const PrimitiveGaussian& ket,
                                       const MagneticField& field) {
    IntegralWorkspace workspace;
    return pair_magnetic_derivatives(
        bra, ket, field, [&](const auto& left, const auto& right) {
            return primitive_overlap(left, right, field, workspace);
        });
}

std::array<Complex, 6>
primitive_kinetic_center_derivatives(const PrimitiveGaussian& bra,
                                     const PrimitiveGaussian& ket,
                                     const MagneticField& field) {
    IntegralWorkspace workspace;
    return pair_center_derivatives(
        bra, ket, field, [&](const auto& left, const auto& right) {
            return primitive_kinetic(left, right, field, workspace);
        });
}

std::array<Complex, 3>
primitive_kinetic_magnetic_derivatives(const PrimitiveGaussian& bra,
                                       const PrimitiveGaussian& ket,
                                       const MagneticField& field) {
    IntegralWorkspace workspace;
    return pair_magnetic_derivatives(
        bra, ket, field, [&](const auto& left, const auto& right) {
            return primitive_kinetic(left, right, field, workspace);
        });
}

std::array<Complex, 6>
primitive_magnetic_kinetic_center_derivatives(const PrimitiveGaussian& bra,
                                              const PrimitiveGaussian& ket,
                                              const MagneticField& field) {
    IntegralWorkspace workspace;
    return pair_center_derivatives(
        bra, ket, field, [&](const auto& left, const auto& right) {
            return primitive_magnetic_kinetic(left, right, field, workspace);
        });
}

std::array<Complex, 3>
primitive_magnetic_kinetic_magnetic_derivatives(const PrimitiveGaussian& bra,
                                                const PrimitiveGaussian& ket,
                                                const MagneticField& field) {
    IntegralWorkspace workspace;
    const auto raw_bra = raw_primitive(bra);
    const auto raw_ket = raw_primitive(ket);
    auto result = pair_magnetic_derivatives(
        raw_bra, raw_ket, field, [&](const auto& left, const auto& right) {
            return primitive_magnetic_kinetic(left, right, field, workspace);
        });
    for (std::size_t axis = 0; axis < axis_count; ++axis) {
        result[axis] += explicit_magnetic_kinetic_response(raw_bra, raw_ket, axis,
                                                           field, workspace);
    }
    return result;
}

std::array<Complex, 6> primitive_nuclear_attraction_center_derivatives(
    const PrimitiveGaussian& bra, const PrimitiveGaussian& ket,
    std::span<const Nucleus> nuclei, const MagneticField& field) {
    NuclearAttractionWorkspace workspace;
    return pair_center_derivatives(
        bra, ket, field, [&](const auto& left, const auto& right) {
            return primitive_nuclear_attraction(left, right, nuclei, field, workspace);
        });
}

std::array<Complex, 3> primitive_nuclear_attraction_magnetic_derivatives(
    const PrimitiveGaussian& bra, const PrimitiveGaussian& ket,
    std::span<const Nucleus> nuclei, const MagneticField& field) {
    NuclearAttractionWorkspace workspace;
    return pair_magnetic_derivatives(
        bra, ket, field, [&](const auto& left, const auto& right) {
            return primitive_nuclear_attraction(left, right, nuclei, field, workspace);
        });
}

void primitive_nuclear_attraction_nucleus_derivatives(const PrimitiveGaussian& bra,
                                                      const PrimitiveGaussian& ket,
                                                      std::span<const Nucleus> nuclei,
                                                      const MagneticField& field,
                                                      std::span<Complex> output) {
    if (output.size() != nuclei.size() * axis_count) {
        throw std::length_error(
            "nucleus-derivative primitive output has an incorrect size");
    }
    const auto raw_bra = raw_primitive(bra);
    const auto raw_ket = raw_primitive(ket);
    NuclearAttractionWorkspace workspace;
    for (std::size_t nucleus = 0; nucleus < nuclei.size(); ++nucleus) {
        const std::span<const Nucleus> single(&nuclei[nucleus], 1U);
        for (std::size_t axis = 0; axis < axis_count; ++axis) {
            output[nucleus * axis_count + axis] =
                spatial_response(raw_bra, true, axis, field, [&](const auto& changed) {
                    return primitive_nuclear_attraction(changed, raw_ket, single, field,
                                                        workspace);
                });
            output[nucleus * axis_count + axis] +=
                spatial_response(raw_ket, false, axis, field, [&](const auto& changed) {
                    return primitive_nuclear_attraction(raw_bra, changed, single, field,
                                                        workspace);
                });
        }
    }
}

std::array<Complex, 12> primitive_eri_center_derivatives(const PrimitiveGaussian& a,
                                                         const PrimitiveGaussian& b,
                                                         const PrimitiveGaussian& c,
                                                         const PrimitiveGaussian& d,
                                                         const MagneticField& field) {
    const std::array<PrimitiveGaussian, 4> raw{raw_primitive(a), raw_primitive(b),
                                               raw_primitive(c), raw_primitive(d)};
    EriWorkspace workspace;
    std::array<Complex, 12> result{};
    for (std::size_t center = 0; center < 4U; ++center) {
        for (std::size_t axis = 0; axis < axis_count; ++axis) {
            result[center * axis_count + axis] = center_response(
                raw[center], center == 0U || center == 2U, axis, field,
                [&](const auto& changed) {
                    auto values = raw;
                    values[center] = changed;
                    return primitive_eri(values[0], values[1], values[2], values[3],
                                         field, workspace);
                });
        }
    }
    return result;
}

std::array<Complex, 3> primitive_eri_magnetic_derivatives(const PrimitiveGaussian& a,
                                                          const PrimitiveGaussian& b,
                                                          const PrimitiveGaussian& c,
                                                          const PrimitiveGaussian& d,
                                                          const MagneticField& field) {
    const std::array<PrimitiveGaussian, 4> raw{raw_primitive(a), raw_primitive(b),
                                               raw_primitive(c), raw_primitive(d)};
    EriWorkspace workspace;
    std::array<Complex, 3> result{};
    for (std::size_t axis = 0; axis < axis_count; ++axis) {
        for (std::size_t center = 0; center < 4U; ++center) {
            result[axis] += magnetic_response(raw[center], center == 0U || center == 2U,
                                              axis, field, [&](const auto& changed) {
                                                  auto values = raw;
                                                  values[center] = changed;
                                                  return primitive_eri(
                                                      values[0], values[1], values[2],
                                                      values[3], field, workspace);
                                              });
        }
    }
    return result;
}

#define GIAO_DEFINE_PAIR_BLOCKS(NAME, PRIMITIVE_CENTER, PRIMITIVE_MAGNETIC)            \
    void compute_##NAME##_center_derivatives(const Shell& a, const Shell& b,           \
                                             const MagneticField& field,               \
                                             std::span<Complex> output) {              \
        compute_pair_block<6U>(a, b, output, [&](const auto& bra, const auto& ket) {   \
            return PRIMITIVE_CENTER(bra, ket, field);                                  \
        });                                                                            \
    }                                                                                  \
    void compute_##NAME##_magnetic_derivatives(const Shell& a, const Shell& b,         \
                                               const MagneticField& field,             \
                                               std::span<Complex> output) {            \
        compute_pair_block<3U>(a, b, output, [&](const auto& bra, const auto& ket) {   \
            return PRIMITIVE_MAGNETIC(bra, ket, field);                                \
        });                                                                            \
    }

GIAO_DEFINE_PAIR_BLOCKS(overlap, primitive_overlap_center_derivatives,
                        primitive_overlap_magnetic_derivatives)
GIAO_DEFINE_PAIR_BLOCKS(kinetic, primitive_kinetic_center_derivatives,
                        primitive_kinetic_magnetic_derivatives)
GIAO_DEFINE_PAIR_BLOCKS(magnetic_kinetic, primitive_magnetic_kinetic_center_derivatives,
                        primitive_magnetic_kinetic_magnetic_derivatives)

#undef GIAO_DEFINE_PAIR_BLOCKS

void compute_nuclear_attraction_center_derivatives(const Shell& a, const Shell& b,
                                                   std::span<const Nucleus> nuclei,
                                                   const MagneticField& field,
                                                   std::span<Complex> output) {
    compute_pair_block<6U>(a, b, output, [&](const auto& bra, const auto& ket) {
        return primitive_nuclear_attraction_center_derivatives(bra, ket, nuclei, field);
    });
}

void compute_nuclear_attraction_magnetic_derivatives(const Shell& a, const Shell& b,
                                                     std::span<const Nucleus> nuclei,
                                                     const MagneticField& field,
                                                     std::span<Complex> output) {
    compute_pair_block<3U>(a, b, output, [&](const auto& bra, const auto& ket) {
        return primitive_nuclear_attraction_magnetic_derivatives(bra, ket, nuclei,
                                                                 field);
    });
}

void compute_nuclear_attraction_nucleus_derivatives(const Shell& a, const Shell& b,
                                                    std::span<const Nucleus> nuclei,
                                                    const MagneticField& field,
                                                    std::span<Complex> output) {
    const std::size_t derivative_count = nuclei.size() * axis_count;
    const std::size_t block_size = shell_pair_size(a, b);
    if (output.size() != derivative_count * block_size) {
        throw std::length_error(
            "nucleus-derivative shell-pair output has an incorrect size");
    }
    std::fill(output.begin(), output.end(), Complex{});
    std::vector<Complex> primitive_values(derivative_count);
    for (std::size_t primitive_a = 0; primitive_a < a.primitive_count();
         ++primitive_a) {
        for (std::size_t primitive_b = 0; primitive_b < b.primitive_count();
             ++primitive_b) {
            for (std::size_t component_a = 0; component_a < a.cartesian_count();
                 ++component_a) {
                const PrimitiveGaussian bra(
                    a.exponents()[primitive_a], a.center(), a.components()[component_a],
                    primitive_normalization(a.exponents()[primitive_a],
                                            a.components()[component_a]),
                    false);
                for (std::size_t component_b = 0; component_b < b.cartesian_count();
                     ++component_b) {
                    const PrimitiveGaussian ket(
                        b.exponents()[primitive_b], b.center(),
                        b.components()[component_b],
                        primitive_normalization(b.exponents()[primitive_b],
                                                b.components()[component_b]),
                        false);
                    primitive_nuclear_attraction_nucleus_derivatives(
                        bra, ket, nuclei, field, primitive_values);
                    for (std::size_t contraction_a = 0;
                         contraction_a < a.contraction_count(); ++contraction_a) {
                        const std::size_t ao_a =
                            contraction_a * a.cartesian_count() + component_a;
                        for (std::size_t contraction_b = 0;
                             contraction_b < b.contraction_count(); ++contraction_b) {
                            const std::size_t ao_b =
                                contraction_b * b.cartesian_count() + component_b;
                            const double weight =
                                a.coefficient(contraction_a, primitive_a) *
                                b.coefficient(contraction_b, primitive_b);
                            const std::size_t element = ao_a * b.ao_count() + ao_b;
                            for (std::size_t derivative = 0;
                                 derivative < derivative_count; ++derivative) {
                                output[derivative * block_size + element] +=
                                    weight * primitive_values[derivative];
                            }
                        }
                    }
                }
            }
        }
    }
}

void compute_eri_center_derivatives(const Shell& a, const Shell& b,
    const Shell& c, const Shell& d, const MagneticField& field, std::span<Complex> output) {
    compute_quartet_block<12U>(a,b,c,d,output,[&](const auto& ga,const auto& gb,const auto& gc,const auto& gd) {
        return primitive_eri_center_derivatives(ga,gb,gc,gd,field);
    });
}

void compute_eri_magnetic_derivatives(const Shell& a, const Shell& b,
    const Shell& c, const Shell& d, const MagneticField& field, std::span<Complex> output) {
    compute_quartet_block<3U>(a,b,c,d,output,[&](const auto& ga,const auto& gb,const auto& gc,const auto& gd) {
        return primitive_eri_magnetic_derivatives(ga,gb,gc,gd,field);
    });
}

void compute_eri_mixed_derivatives(const Shell& a, const Shell& b,
    const Shell& c, const Shell& d, const MagneticField& field, std::span<Complex> output) {
    compute_quartet_block<36U>(a,b,c,d,output,[&](const auto& ga,const auto& gb,const auto& gc,const auto& gd) {
        return primitive_eri_mixed_derivatives(ga,gb,gc,gd,field);
    });
}

#define GIAO_DEFINE_BASIS_DERIVATIVES(NAME)                                            \
    void compute_##NAME##_nuclear_derivative_matrices(                                 \
        const Basis& basis, const MagneticField& field, std::span<Complex> output) {   \
        compute_basis_nuclear_derivatives(                                             \
            basis, output, [&](const auto& a, const auto& b, auto& block) {            \
                compute_##NAME##_center_derivatives(a, b, field, block);               \
            });                                                                        \
    }                                                                                  \
    void compute_##NAME##_magnetic_derivative_matrices(                                \
        const Basis& basis, const MagneticField& field, std::span<Complex> output) {   \
        compute_basis_magnetic_derivatives(                                            \
            basis, output, [&](const auto& a, const auto& b, auto& block) {            \
                compute_##NAME##_magnetic_derivatives(a, b, field, block);             \
            });                                                                        \
    }

GIAO_DEFINE_BASIS_DERIVATIVES(overlap)
GIAO_DEFINE_BASIS_DERIVATIVES(kinetic)
GIAO_DEFINE_BASIS_DERIVATIVES(magnetic_kinetic)

#undef GIAO_DEFINE_BASIS_DERIVATIVES

void compute_nuclear_attraction_nuclear_derivative_matrices(
    const Basis& basis, std::span<const Nucleus> nuclei, const MagneticField& field,
    std::span<Complex> shell_center_output, std::span<Complex> nucleus_center_output) {
    compute_basis_nuclear_derivatives(
        basis, shell_center_output, [&](const auto& a, const auto& b, auto& block) {
            compute_nuclear_attraction_center_derivatives(a, b, nuclei, field, block);
        });

    const std::size_t count = basis.ao_count();
    const std::size_t derivative_count = nuclei.size() * axis_count;
    if (nucleus_center_output.size() != derivative_count * count * count) {
        throw std::length_error(
            "nucleus-derivative basis output has an incorrect size");
    }
    std::fill(nucleus_center_output.begin(), nucleus_center_output.end(), Complex{});
    std::vector<Complex> block;
    for (std::size_t shell_a = 0; shell_a < basis.shells().size(); ++shell_a) {
        const auto& a = basis.shells()[shell_a];
        for (std::size_t shell_b = 0; shell_b < basis.shells().size(); ++shell_b) {
            const auto& b = basis.shells()[shell_b];
            const std::size_t pair_size = shell_pair_size(a, b);
            block.resize(derivative_count * pair_size);
            compute_nuclear_attraction_nucleus_derivatives(a, b, nuclei, field, block);
            for (std::size_t derivative = 0; derivative < derivative_count;
                 ++derivative) {
                for (std::size_t ao_a = 0; ao_a < a.ao_count(); ++ao_a) {
                    for (std::size_t ao_b = 0; ao_b < b.ao_count(); ++ao_b) {
                        const std::size_t global_a = basis.ao_offsets()[shell_a] + ao_a;
                        const std::size_t global_b = basis.ao_offsets()[shell_b] + ao_b;
                        nucleus_center_output[(derivative * count + global_a) * count +
                                              global_b] =
                            block[derivative * pair_size + ao_a * b.ao_count() + ao_b];
                    }
                }
            }
        }
    }
}

void compute_nuclear_attraction_magnetic_derivative_matrices(
    const Basis& basis, std::span<const Nucleus> nuclei, const MagneticField& field,
    std::span<Complex> output) {
    compute_basis_magnetic_derivatives(
        basis, output, [&](const auto& a, const auto& b, auto& block) {
            compute_nuclear_attraction_magnetic_derivatives(a, b, nuclei, field, block);
        });
}

std::array<Complex, 6> primitive_gradient_center_derivatives(
    const PrimitiveGaussian& bra, const PrimitiveGaussian& ket,
    Axis axis, const MagneticField& field) {
    IntegralWorkspace workspace;
    return pair_center_derivatives(bra, ket, field, [&](const auto& a, const auto& b) {
        return primitive_gradient(a, b, axis, field, workspace);
    });
}

void compute_gradient_center_derivatives(const Shell& a, const Shell& b,
    Axis axis, const MagneticField& field, std::span<Complex> output) {
    compute_pair_block<6U>(a, b, output, [&](const auto& bra, const auto& ket) {
        return primitive_gradient_center_derivatives(bra, ket, axis, field);
    });
}

void compute_gradient_nuclear_derivative_matrices(const Basis& basis,
    Axis axis, const MagneticField& field, std::span<Complex> output,
    std::size_t threads) {
    property_derivative_matrices(basis, 0U, output, {}, threads,
        [&](const Shell& a, const Shell& b, std::span<Complex> block) {
            compute_gradient_center_derivatives(a, b, axis, field, block);
        });
}

std::array<Complex, 3> primitive_gradient_magnetic_derivatives(
    const PrimitiveGaussian& bra, const PrimitiveGaussian& ket,
    Axis axis, const MagneticField& field) {
    IntegralWorkspace workspace;
    return pair_magnetic_derivatives(bra, ket, field, [&](const auto& a, const auto& b) {
        return primitive_gradient(a, b, axis, field, workspace);
    });
}

void compute_gradient_magnetic_derivatives(const Shell& a, const Shell& b,
    Axis axis, const MagneticField& field, std::span<Complex> output) {
    compute_pair_block<3U>(a, b, output, [&](const auto& bra, const auto& ket) {
        return primitive_gradient_magnetic_derivatives(bra, ket, axis, field);
    });
}

void compute_gradient_magnetic_derivative_matrices(const Basis& basis,
    Axis axis, const MagneticField& field, std::span<Complex> output,
    std::size_t threads) {
    property_magnetic_matrices(basis, output, threads,
        [&](const Shell& a, const Shell& b, std::span<Complex> block) {
            compute_gradient_magnetic_derivatives(a, b, axis, field, block);
        });
}

std::array<Complex, 6> primitive_momentum_center_derivatives(
    const PrimitiveGaussian& bra, const PrimitiveGaussian& ket,
    Axis axis, const MagneticField& field) {
    IntegralWorkspace workspace;
    return pair_center_derivatives(bra, ket, field, [&](const auto& a, const auto& b) {
        return primitive_momentum(a, b, axis, field, workspace);
    });
}

void compute_momentum_center_derivatives(const Shell& a, const Shell& b,
    Axis axis, const MagneticField& field, std::span<Complex> output) {
    compute_pair_block<6U>(a, b, output, [&](const auto& bra, const auto& ket) {
        return primitive_momentum_center_derivatives(bra, ket, axis, field);
    });
}

void compute_momentum_nuclear_derivative_matrices(const Basis& basis,
    Axis axis, const MagneticField& field, std::span<Complex> output,
    std::size_t threads) {
    property_derivative_matrices(basis, 0U, output, {}, threads,
        [&](const Shell& a, const Shell& b, std::span<Complex> block) {
            compute_momentum_center_derivatives(a, b, axis, field, block);
        });
}

std::array<Complex, 3> primitive_momentum_magnetic_derivatives(
    const PrimitiveGaussian& bra, const PrimitiveGaussian& ket,
    Axis axis, const MagneticField& field) {
    IntegralWorkspace workspace;
    return pair_magnetic_derivatives(bra, ket, field, [&](const auto& a, const auto& b) {
        return primitive_momentum(a, b, axis, field, workspace);
    });
}

void compute_momentum_magnetic_derivatives(const Shell& a, const Shell& b,
    Axis axis, const MagneticField& field, std::span<Complex> output) {
    compute_pair_block<3U>(a, b, output, [&](const auto& bra, const auto& ket) {
        return primitive_momentum_magnetic_derivatives(bra, ket, axis, field);
    });
}

void compute_momentum_magnetic_derivative_matrices(const Basis& basis,
    Axis axis, const MagneticField& field, std::span<Complex> output,
    std::size_t threads) {
    property_magnetic_matrices(basis, output, threads,
        [&](const Shell& a, const Shell& b, std::span<Complex> block) {
            compute_momentum_magnetic_derivatives(a, b, axis, field, block);
        });
}

std::array<Complex, 3> primitive_moment_magnetic_derivatives(
    const PrimitiveGaussian& bra, const PrimitiveGaussian& ket,
    const CartesianMoment& moment, const MagneticField& field) {
    IntegralWorkspace workspace;
    return pair_magnetic_derivatives(bra, ket, field, [&](const auto& a, const auto& b) {
        return primitive_moment(a, b, moment, field, workspace);
    });
}

void compute_moment_magnetic_derivatives(const Shell& a, const Shell& b,
    const CartesianMoment& moment, const MagneticField& field, std::span<Complex> output) {
    compute_pair_block<3U>(a, b, output, [&](const auto& bra, const auto& ket) {
        return primitive_moment_magnetic_derivatives(bra, ket, moment, field);
    });
}

void compute_moment_magnetic_derivative_matrices(const Basis& basis,
    const CartesianMoment& moment, const MagneticField& field, std::span<Complex> output,
    std::size_t threads) {
    property_magnetic_matrices(basis, output, threads,
        [&](const Shell& a, const Shell& b, std::span<Complex> block) {
            compute_moment_magnetic_derivatives(a, b, moment, field, block);
        });
}

std::array<Complex,18> primitive_overlap_mixed_derivatives(
    const PrimitiveGaussian& bra, const PrimitiveGaussian& ket,
    const MagneticField& field) {
    IntegralWorkspace workspace;
    return pair_mixed_derivatives(bra,ket,field,
        [&](const auto& a,const auto& b) { return primitive_overlap(a,b,field,workspace); },
        [&](const auto& a,const auto& b) { return primitive_overlap_magnetic_derivatives(a,b,field); });
}

void compute_overlap_mixed_derivatives(const Shell& a, const Shell& b,
    const MagneticField& field, std::span<Complex> output) {
    compute_pair_block<18U>(a,b,output,[&](const auto& bra,const auto& ket) {
        return primitive_overlap_mixed_derivatives(bra,ket,field);
    });
}

void compute_overlap_mixed_derivative_matrices(const Basis& basis,
    const MagneticField& field, std::span<Complex> output, std::size_t threads) {
    property_derivative_matrices(basis,0U,output,{},threads,
        [&](const auto& a,const auto& b,std::span<Complex> block) { compute_overlap_mixed_derivatives(a,b,field,block); },9U);
}

std::array<Complex,18> primitive_kinetic_mixed_derivatives(
    const PrimitiveGaussian& bra, const PrimitiveGaussian& ket,
    const MagneticField& field) {
    IntegralWorkspace workspace;
    return pair_mixed_derivatives(bra,ket,field,
        [&](const auto& a,const auto& b) { return primitive_kinetic(a,b,field,workspace); },
        [&](const auto& a,const auto& b) { return primitive_kinetic_magnetic_derivatives(a,b,field); });
}

void compute_kinetic_mixed_derivatives(const Shell& a, const Shell& b,
    const MagneticField& field, std::span<Complex> output) {
    compute_pair_block<18U>(a,b,output,[&](const auto& bra,const auto& ket) {
        return primitive_kinetic_mixed_derivatives(bra,ket,field);
    });
}

void compute_kinetic_mixed_derivative_matrices(const Basis& basis,
    const MagneticField& field, std::span<Complex> output, std::size_t threads) {
    property_derivative_matrices(basis,0U,output,{},threads,
        [&](const auto& a,const auto& b,std::span<Complex> block) { compute_kinetic_mixed_derivatives(a,b,field,block); },9U);
}

std::array<Complex,18> primitive_magnetic_kinetic_mixed_derivatives(
    const PrimitiveGaussian& bra, const PrimitiveGaussian& ket,
    const MagneticField& field) {
    IntegralWorkspace workspace;
    return pair_mixed_derivatives(bra,ket,field,
        [&](const auto& a,const auto& b) { return primitive_magnetic_kinetic(a,b,field,workspace); },
        [&](const auto& a,const auto& b) { return primitive_magnetic_kinetic_magnetic_derivatives(a,b,field); });
}

void compute_magnetic_kinetic_mixed_derivatives(const Shell& a, const Shell& b,
    const MagneticField& field, std::span<Complex> output) {
    compute_pair_block<18U>(a,b,output,[&](const auto& bra,const auto& ket) {
        return primitive_magnetic_kinetic_mixed_derivatives(bra,ket,field);
    });
}

void compute_magnetic_kinetic_mixed_derivative_matrices(const Basis& basis,
    const MagneticField& field, std::span<Complex> output, std::size_t threads) {
    property_derivative_matrices(basis,0U,output,{},threads,
        [&](const auto& a,const auto& b,std::span<Complex> block) { compute_magnetic_kinetic_mixed_derivatives(a,b,field,block); },9U);
}

std::array<Complex,18> primitive_nuclear_attraction_mixed_derivatives(
    const PrimitiveGaussian& bra, const PrimitiveGaussian& ket,
    std::span<const Nucleus> nuclei, const MagneticField& field) {
    NuclearAttractionWorkspace workspace;
    return pair_mixed_derivatives(bra,ket,field,
        [&](const auto& a,const auto& b) { return primitive_nuclear_attraction(a,b,nuclei, field,workspace); },
        [&](const auto& a,const auto& b) { return primitive_nuclear_attraction_magnetic_derivatives(a,b,nuclei, field); });
}

void compute_nuclear_attraction_mixed_derivatives(const Shell& a, const Shell& b,
    std::span<const Nucleus> nuclei, const MagneticField& field, std::span<Complex> output) {
    compute_pair_block<18U>(a,b,output,[&](const auto& bra,const auto& ket) {
        return primitive_nuclear_attraction_mixed_derivatives(bra,ket,nuclei, field);
    });
}

std::array<Complex,36> primitive_eri_mixed_derivatives(
    const PrimitiveGaussian& a, const PrimitiveGaussian& b, const PrimitiveGaussian& c,
    const PrimitiveGaussian& d, const MagneticField& field) {
    const std::array<PrimitiveGaussian,4> raw{raw_primitive(a),raw_primitive(b),raw_primitive(c),raw_primitive(d)};
    EriWorkspace workspace;
    std::array<Complex,36> result{};
    for (std::size_t center=0; center<4U; ++center) {
        for (std::size_t r=0; r<3U; ++r) {
            for (std::size_t axis=0; axis<3U; ++axis) {
                auto value = [&](const auto& changed) {
                    auto p=raw; p[center]=changed;
                    return primitive_eri(p[0],p[1],p[2],p[3],field,workspace);
                };
                auto magnetic = [&](const auto& changed) {
                    auto p=raw; p[center]=changed;
                    return primitive_eri_magnetic_derivatives(p[0],p[1],p[2],p[3],field)[axis];
                };
                result[(center*3U+r)*3U+axis] = center_response(raw[center],center%2U==0U,r,field,magnetic) +
                    center_phase_field_response(raw[center],center%2U==0U,r,axis,value);
            }
        }
    }
    return result;
}

void primitive_nuclear_attraction_nucleus_mixed_derivatives(const PrimitiveGaussian& bra,
    const PrimitiveGaussian& ket, std::span<const Nucleus> nuclei,
    const MagneticField& field, std::span<Complex> output) {
    if (output.size()!=nuclei.size()*9U) {
        throw std::length_error("nucleus mixed output has an incorrect size");
    }
    for (std::size_t nucleus=0; nucleus<nuclei.size(); ++nucleus) {
        const std::span<const Nucleus> single(&nuclei[nucleus],1U);
        for (std::size_t axis=0; axis<3U; ++axis) {
            const auto response=pair_magnetic_derivatives(bra,ket,field,
                [&](const auto& a,const auto& b) {
                    std::array<Complex,3> derivatives{};
                    primitive_nuclear_attraction_nucleus_derivatives(a,b,single,field,derivatives);
                    return derivatives[axis];
                });
            for (std::size_t b=0; b<3U; ++b) {
                output[(nucleus*3U+axis)*3U+b]=response[b];
            }
        }
    }
}

void compute_nuclear_attraction_nucleus_mixed_derivatives(const Shell& a, const Shell& b,
    std::span<const Nucleus> nuclei, const MagneticField& field, std::span<Complex> output) {
    const auto size=shell_pair_size(a,b);
    if (output.size()!=nuclei.size()*9U*size) {
        throw std::length_error("nucleus mixed shell output has an incorrect size");
    }
    for (std::size_t nucleus=0; nucleus<nuclei.size(); ++nucleus) {
        const std::span<const Nucleus> single(&nuclei[nucleus],1U);
        compute_pair_block<9U>(a,b,output.subspan(nucleus*9U*size,9U*size),[&](const auto& bra,const auto& ket) {
            std::array<Complex,9> result{};
            primitive_nuclear_attraction_nucleus_mixed_derivatives(bra,ket,single,field,result);
            return result;
        });
    }
}

void compute_nuclear_attraction_mixed_derivative_matrices(const Basis& basis,
    std::span<const Nucleus> nuclei, const MagneticField& field,
    std::span<Complex> centers, std::span<Complex> potential, std::size_t threads) {
    property_derivative_matrices(basis,nuclei.size()*9U,centers,potential,threads,
        [&](const Shell& a,const Shell& b,std::span<Complex> block) {
            const auto size=shell_pair_size(a,b);
            compute_nuclear_attraction_mixed_derivatives(a,b,nuclei,field,block.first(18U*size));
            compute_nuclear_attraction_nucleus_mixed_derivatives(a,b,nuclei,field,block.subspan(18U*size));
        },9U);
}

} // namespace giao

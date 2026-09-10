#include "giao_integrals/boys.hpp"
#include "giao_integrals/eri.hpp"
#include "giao_integrals/overlap.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>

namespace {

struct OsWorkspace {
    std::vector<giao::Complex> values;
    std::vector<std::uint32_t> tags;
    std::vector<giao::Complex> boys;
    std::uint32_t generation{};
};

giao::Complex os_eri(const std::array<giao::PrimitiveGaussian, 4>& primitives,
                     const giao::MagneticField& field, OsWorkspace& workspace) {
    const auto pair_ab = giao::gaussian_product(primitives[0], primitives[1], field);
    const auto pair_cd = giao::gaussian_product(primitives[2], primitives[3], field);
    const double p = pair_ab.exponent;
    const double q = pair_cd.exponent;
    const double rho = p * q / (p + q);
    const std::array<giao::Complex, 3> displacement{
        pair_ab.complex_center[0] - pair_cd.complex_center[0],
        pair_ab.complex_center[1] - pair_cd.complex_center[1],
        pair_ab.complex_center[2] - pair_cd.complex_center[2]};
    const auto argument =
        rho * (displacement[0] * displacement[0] +
               displacement[1] * displacement[1] +
               displacement[2] * displacement[2]);
    std::array<giao::Complex, 3> weighted_center{};
    for (std::size_t axis = 0; axis < 3; ++axis) {
        weighted_center[axis] =
            (p * pair_ab.complex_center[axis] +
             q * pair_cd.complex_center[axis]) /
            (p + q);
    }

    std::array<std::uint16_t, 12> target{};
    std::size_t maximum_order = 0U;
    for (std::size_t center = 0; center < 4; ++center) {
        for (std::size_t axis = 0; axis < 3; ++axis) {
            target[3U * center + axis] = primitives[center].angular[axis];
            maximum_order += primitives[center].angular[axis];
        }
    }
    std::size_t table_size = maximum_order + 1U;
    for (const auto extent : target) {
        table_size *= static_cast<std::size_t>(extent) + 1U;
    }
    workspace.values.resize(std::max(workspace.values.size(), table_size));
    workspace.tags.resize(std::max(workspace.tags.size(), table_size));
    workspace.boys.resize(std::max(workspace.boys.size(), maximum_order + 1U));
    ++workspace.generation;
    if (workspace.generation == 0U) {
        std::fill(workspace.tags.begin(), workspace.tags.end(), 0U);
        ++workspace.generation;
    }
    const auto generation = workspace.generation;
    [[maybe_unused]] const auto diagnostics = giao::compute_boys(
        argument,
        std::span<giao::Complex>(workspace.boys.data(), maximum_order + 1U));

    const auto index_of = [&](std::size_t order,
                              const std::array<std::uint16_t, 12>& state) {
        std::size_t index = order;
        for (std::size_t position = 0; position < state.size(); ++position) {
            index = index * (static_cast<std::size_t>(target[position]) + 1U) +
                    state[position];
        }
        return index;
    };
    std::function<giao::Complex(std::array<std::uint16_t, 12>, std::size_t)>
        integral;
    integral = [&](std::array<std::uint16_t, 12> state,
                   std::size_t order) -> giao::Complex {
        const auto index = index_of(order, state);
        if (workspace.tags[index] == generation) {
            return workspace.values[index];
        }
        if (std::all_of(state.begin(), state.end(),
                        [](auto value) { return value == 0U; })) {
            workspace.values[index] = workspace.boys[order];
            workspace.tags[index] = generation;
            return workspace.values[index];
        }
        for (std::size_t center = 0; center < 4; ++center) {
            for (std::size_t axis = 0; axis < 3; ++axis) {
                const auto position = 3U * center + axis;
                if (state[position] == 0U) {
                    continue;
                }
                auto parent = state;
                --parent[position];
                const bool first_pair = center < 2U;
                const auto& pair = first_pair ? pair_ab : pair_cd;
                const double pair_exponent = first_pair ? p : q;
                const auto real_center = primitives[center].center[axis];
                const auto complex_center = pair.complex_center[axis];
                giao::Complex result =
                    (complex_center - real_center) * integral(parent, order);
                result += (weighted_center[axis] - complex_center) *
                          integral(parent, order + 1U);
                const std::size_t same_begin = first_pair ? 0U : 2U;
                const std::size_t other_begin = first_pair ? 2U : 0U;
                for (std::size_t other = same_begin; other < same_begin + 2U;
                     ++other) {
                    const auto other_position = 3U * other + axis;
                    if (parent[other_position] == 0U) {
                        continue;
                    }
                    auto reduced = parent;
                    const double coefficient = reduced[other_position];
                    --reduced[other_position];
                    result += coefficient / (2.0 * pair_exponent) *
                              (integral(reduced, order) -
                               rho / pair_exponent *
                                   integral(reduced, order + 1U));
                }
                for (std::size_t other = other_begin;
                     other < other_begin + 2U; ++other) {
                    const auto other_position = 3U * other + axis;
                    if (parent[other_position] == 0U) {
                        continue;
                    }
                    auto reduced = parent;
                    const double coefficient = reduced[other_position];
                    --reduced[other_position];
                    result += coefficient / (2.0 * (p + q)) *
                              integral(reduced, order + 1U);
                }
                workspace.values[index] = result;
                workspace.tags[index] = generation;
                return result;
            }
        }
        return {};
    };

    auto result = integral(target, 0U);
    result *= 2.0 * std::pow(std::numbers::pi, 2.5) /
              (p * q * std::sqrt(p + q)) * pair_ab.prefactor() *
              pair_cd.prefactor();
    for (const auto& primitive : primitives) {
        result *= primitive.coefficient;
        if (primitive.normalized) {
            result *= giao::primitive_normalization(primitive.exponent,
                                                     primitive.angular);
        }
    }
    return result;
}

template <typename Function>
std::pair<double, giao::Complex> measure(Function&& function,
                                         std::size_t iterations) {
    giao::Complex checksum{};
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        checksum += function();
    }
    const auto stop = std::chrono::steady_clock::now();
    return {std::chrono::duration<double>(stop - start).count(), checksum};
}

void run_case(const std::string& name,
              const std::array<giao::PrimitiveGaussian, 4>& primitives,
              const giao::MagneticField& field, std::size_t iterations) {
    giao::EriWorkspace md_workspace;
    OsWorkspace os_workspace;
    const auto md_value = giao::primitive_eri(
        primitives[0], primitives[1], primitives[2], primitives[3], field,
        md_workspace);
    const auto os_value = os_eri(primitives, field, os_workspace);
    const auto [md_seconds, md_checksum] = measure(
        [&] {
            return giao::primitive_eri(primitives[0], primitives[1],
                                       primitives[2], primitives[3], field,
                                       md_workspace);
        },
        iterations);
    const auto [os_seconds, os_checksum] =
        measure([&] { return os_eri(primitives, field, os_workspace); }, iterations);
    std::cout << std::setprecision(12)
              << "{\"schema_version\":1,\"benchmark\":\"eri_backend\""
              << ",\"case\":\"" << name << "\",\"iterations\":"
              << iterations << ",\"md_seconds\":" << md_seconds
              << ",\"os_seconds\":" << os_seconds
              << ",\"os_over_md\":" << os_seconds / md_seconds
              << ",\"absolute_difference\":" << std::abs(md_value - os_value)
              << ",\"md_checksum_real\":" << md_checksum.real()
              << ",\"os_checksum_real\":" << os_checksum.real() << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t scale = 1U;
    if (argc == 2 && std::string(argv[1]) == "--quick") {
        scale = 10U;
    }
    const std::array centers{
        giao::Vec3{-0.3, 0.2, 0.5}, giao::Vec3{0.6, -0.4, 0.1},
        giao::Vec3{0.2, 0.7, -0.6}, giao::Vec3{-0.5, -0.1, 0.4}};
    const std::array exponents{0.7, 1.2, 0.9, 1.5};
    const std::array fields{
        std::pair{std::string("zero"), giao::MagneticField{}},
        std::pair{std::string("finite"),
                  giao::MagneticField({0.7, -0.4, 0.3}, {0.1, 0.2, -0.5})}};
    const std::array angular_cases{
        std::pair{std::string("ssss"),
                  std::array<giao::CartesianExponent, 4>{
                      giao::CartesianExponent{}, {}, {}, {}}},
        std::pair{std::string("pppp"),
                  std::array<giao::CartesianExponent, 4>{
                      giao::CartesianExponent{1, 0, 0}, {0, 1, 0},
                      {0, 0, 1}, {1, 0, 0}}},
        std::pair{std::string("dpps"),
                  std::array<giao::CartesianExponent, 4>{
                      giao::CartesianExponent{2, 0, 0}, {0, 0, 1},
                      {0, 1, 0}, {}}}};
    for (const auto& [field_name, field] : fields) {
        for (const auto& [angular_name, angular] : angular_cases) {
            std::array<giao::PrimitiveGaussian, 4> primitives{
                giao::PrimitiveGaussian(exponents[0], centers[0], angular[0]),
                giao::PrimitiveGaussian(exponents[1], centers[1], angular[1]),
                giao::PrimitiveGaussian(exponents[2], centers[2], angular[2]),
                giao::PrimitiveGaussian(exponents[3], centers[3], angular[3])};
            const std::size_t iterations =
                (angular_name == "ssss" ? 20000U : 4000U) / scale;
            run_case(angular_name + "_" + field_name, primitives, field,
                     iterations);
        }
    }
}

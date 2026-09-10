#include "giao_integrals/eri.hpp"

#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Checksum {
    giao::Complex value{};
    std::size_t integral_count{};
};

void consume(const giao::ShellQuartetBlockView& block, void* user_data) {
    auto& checksum = *static_cast<Checksum*>(user_data);
    for (const auto value : block.values) {
        checksum.value += value;
    }
    checksum.integral_count += block.values.size();
}

std::vector<giao::ShellQuartetIndex> canonical_quartets(std::size_t count) {
    std::vector<giao::ShellQuartetIndex> result;
    for (std::size_t a = 0; a < count; ++a) {
        for (std::size_t b = 0; b < count; ++b) {
            for (std::size_t c = 0; c < count; ++c) {
                for (std::size_t d = 0; d < count; ++d) {
                    const giao::ShellQuartetIndex quartet{
                        static_cast<std::uint32_t>(a),
                        static_cast<std::uint32_t>(b),
                        static_cast<std::uint32_t>(c),
                        static_cast<std::uint32_t>(d)};
                    if (giao::canonicalize_shell_quartet(quartet)
                            .shells.as_array() == quartet.as_array()) {
                        result.push_back(quartet);
                    }
                }
            }
        }
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    if (!giao::openmp_enabled()) {
        std::cerr << "OpenMP support is not enabled\n";
        return 2;
    }
    std::size_t iterations = 3U;
    if (argc == 2 && std::string(argv[1]) == "--quick") {
        iterations = 1U;
    }
    const giao::Basis basis({
        giao::Shell({-1.0, 0.0, 0.2}, 1, {1.4, 0.4}, {0.35, 0.72}),
        giao::Shell({0.0, 0.9, -0.3}, 1, {1.2, 0.35}, {-0.2, 0.81}),
        giao::Shell({0.8, -0.4, 0.1}, 1, {1.6, 0.5}, {0.28, 0.76}),
        giao::Shell({-0.3, -0.8, 0.5}, 1, {1.1, 0.3}, {0.42, 0.67}),
    });
    const giao::MagneticField field({0.13, -0.21, 0.17}, {0.2, 0.1, -0.3});
    const auto quartets = canonical_quartets(basis.shells().size());
    const std::array<std::size_t, 4> requested_threads{1U, 2U, 4U, 8U};
    double serial_seconds = 0.0;

    for (const auto threads : requested_threads) {
        if (threads > giao::openmp_max_threads()) {
            continue;
        }
        Checksum checksum;
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
            [[maybe_unused]] const auto statistics =
                giao::evaluate_eri_shell_quartets(
                    basis, quartets, field, {0.0, threads}, nullptr, consume,
                    &checksum);
        }
        const auto stop = std::chrono::steady_clock::now();
        const double seconds =
            std::chrono::duration<double>(stop - start).count();
        if (threads == 1U) {
            serial_seconds = seconds;
        }
        std::cout << std::setprecision(12)
                  << "{\"schema_version\":1,\"benchmark\":\"eri_parallel\""
                  << ",\"shells\":4,\"shell_class\":\"p-p-p-p\""
                  << ",\"primitive_shape\":\"2x2x2x2\""
                  << ",\"field\":\"finite\",\"threads\":" << threads
                  << ",\"iterations\":" << iterations
                  << ",\"quartets_per_iteration\":" << quartets.size()
                  << ",\"integrals_per_iteration\":"
                  << checksum.integral_count / iterations
                  << ",\"seconds\":" << seconds
                  << ",\"speedup\":" << serial_seconds / seconds
                  << ",\"checksum_real\":" << checksum.value.real()
                  << ",\"checksum_imag\":" << checksum.value.imag()
                  << "}\n";
    }
}

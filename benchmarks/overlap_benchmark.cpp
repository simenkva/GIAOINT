#include "giao_integrals/overlap.hpp"

#include <chrono>
#include <complex>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    const giao::Shell a({-0.6, 0.1, 0.3}, 3,
                        {12.0, 3.1, 0.82, 0.21},
                        {0.04, 0.19, 0.51, 0.38});
    const giao::Shell b({0.7, -0.4, 0.2}, 3,
                        {9.0, 2.4, 0.65, 0.17},
                        {-0.03, 0.22, 0.48, 0.41});
    const giao::MagneticField field({0.13, -0.21, 0.17}, {0.2, 0.1, -0.3});
    std::vector<giao::Complex> output(giao::shell_pair_size(a, b));
    giao::IntegralWorkspace workspace;

    constexpr std::size_t iterations = 20000;
    giao::Complex checksum{};
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        giao::compute_overlap(a, b, field, output, workspace);
        checksum += output[iteration % output.size()];
    }
    const auto stop = std::chrono::steady_clock::now();
    const double seconds =
        std::chrono::duration<double>(stop - start).count();
    const double blocks_per_second = static_cast<double>(iterations) / seconds;
    const double integrals_per_second =
        blocks_per_second * static_cast<double>(output.size());

    std::cout << std::setprecision(8) << "shell=f-f primitives=4x4 field=finite "
              << "iterations=" << iterations << " seconds=" << seconds
              << " blocks_per_second=" << blocks_per_second
              << " integrals_per_second=" << integrals_per_second
              << " checksum=" << checksum << '\n';
}

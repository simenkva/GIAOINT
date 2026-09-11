#include "giao_integrals/eri.hpp"
#include "giao_integrals/nuclear.hpp"
#include "giao_integrals/overlap.hpp"
#include "giao_integrals/property.hpp"
#include "giao_integrals/version.hpp"

#include <chrono>
#include <complex>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Measurement {
    std::size_t iterations{};
    double seconds{};
    giao::Complex checksum{};
};

template <typename Function>
Measurement measure(Function&& function, std::span<const giao::Complex> output,
                    double minimum_seconds) {
    function();
    std::size_t iterations = 1U;
    Measurement result;
    do {
        giao::Complex checksum{};
        const auto start = Clock::now();
        for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
            function();
            checksum += output[iteration % output.size()];
        }
        const auto stop = Clock::now();
        result = {iterations,
                  std::chrono::duration<double>(stop - start).count(), checksum};
        iterations *= 2U;
    } while (result.seconds < minimum_seconds);
    return result;
}

const char* compiler_name() {
#if defined(__clang__)
    return "clang";
#elif defined(__GNUC__)
    return "gcc";
#elif defined(_MSC_VER)
    return "msvc";
#else
    return "unknown";
#endif
}

void report(const std::string& name, const std::string& operation,
            const std::string& field_name, const std::string& shell_class,
            const std::string& primitive_shape, std::size_t integral_count,
            const Measurement& measurement) {
    const auto blocks_per_second =
        static_cast<double>(measurement.iterations) / measurement.seconds;
    std::cout << std::setprecision(12)
              << "{\"schema_version\":1,\"library_version\":\"" << giao::version
              << "\""
              << ",\"compiler\":\"" << compiler_name() << "\""
              << ",\"case\":\"" << name << "\""
              << ",\"operator\":\"" << operation << "\""
              << ",\"field\":\"" << field_name << "\""
              << ",\"shell_class\":\"" << shell_class << "\""
              << ",\"primitive_shape\":\"" << primitive_shape << "\""
              << ",\"threads\":1"
              << ",\"iterations\":" << measurement.iterations
              << ",\"seconds\":" << measurement.seconds
              << ",\"blocks_per_second\":" << blocks_per_second
              << ",\"integrals_per_second\":"
              << blocks_per_second * static_cast<double>(integral_count)
              << ",\"checksum_real\":" << measurement.checksum.real()
              << ",\"checksum_imag\":" << measurement.checksum.imag()
              << "}\n";
}

void run_pair_cases(const giao::Shell& a, const giao::Shell& b,
                    const std::string& shell_class,
                    const std::string& primitive_shape,
                    const giao::MagneticField& field,
                    const std::string& field_name, double minimum_seconds) {
    std::vector<giao::Complex> output(giao::shell_pair_size(a, b));
    giao::IntegralWorkspace workspace;
    report("overlap_" + shell_class + "_" + field_name, "overlap", field_name,
           shell_class, primitive_shape, output.size(),
           measure([&] { giao::compute_overlap(a, b, field, output, workspace); },
                   output, minimum_seconds));
    report("kinetic_" + shell_class + "_" + field_name, "kinetic", field_name,
           shell_class, primitive_shape, output.size(),
           measure([&] { giao::compute_kinetic(a, b, field, output, workspace); },
                   output, minimum_seconds));

    const std::vector<giao::Nucleus> nuclei{{1.0, {-0.5, 0.2, 0.1}},
                                             {2.0, {0.6, -0.3, 0.4}}};
    giao::NuclearAttractionWorkspace nuclear_workspace;
    report(
        "attraction_" + shell_class + "_" + field_name,
        "nuclear_attraction", field_name, shell_class, primitive_shape,
        output.size(),
        measure(
            [&] {
                giao::compute_nuclear_attraction(a, b, nuclei, field, output,
                                                  nuclear_workspace);
            },
            output, minimum_seconds));
}

void run_eri_case(const giao::Shell& a, const giao::Shell& b,
                  const giao::Shell& c, const giao::Shell& d,
                  const std::string& shell_class,
                  const std::string& primitive_shape,
                  const giao::MagneticField& field,
                  const std::string& field_name, double minimum_seconds) {
    std::vector<giao::Complex> output(giao::shell_quartet_size(a, b, c, d));
    giao::EriWorkspace workspace;
    report("eri_" + shell_class + "_" + field_name, "eri", field_name,
           shell_class, primitive_shape, output.size(),
           measure(
               [&] { giao::compute_eri(a, b, c, d, field, output, workspace); },
               output, minimum_seconds));
}

}  // namespace

int main(int argc, char** argv) {
    double minimum_seconds = 0.25;
    const std::string mode = argc == 2 ? argv[1] : "";
    if (argc == 2 && std::string(argv[1]) == "--quick") {
        minimum_seconds = 0.02;
    }

    const giao::Shell s_a({-0.6, 0.1, 0.3}, 0, {2.1, 0.55}, {0.3, 0.8});
    const giao::Shell s_b({0.7, -0.4, 0.2}, 0, {1.7, 0.42}, {-0.2, 0.9});
    const giao::Shell p_a({-0.6, 0.1, 0.3}, 1, {2.1, 0.55}, {0.3, 0.8});
    const giao::Shell p_b({0.7, -0.4, 0.2}, 1, {1.7, 0.42}, {-0.2, 0.9});
    const giao::Shell d_a({-0.6, 0.1, 0.3}, 2,
                          {6.2, 1.4, 0.32}, {0.08, 0.42, 0.63});
    const giao::Shell d_b({0.7, -0.4, 0.2}, 2,
                          {5.1, 1.1, 0.27}, {-0.05, 0.47, 0.59});
    const std::array fields{
        std::pair{std::string("zero"), giao::MagneticField{}},
        std::pair{std::string("finite"),
                  giao::MagneticField({0.13, -0.21, 0.17},
                                      {0.2, 0.1, -0.3})}};

    for (const auto& [field_name, field] : fields) {
        // Four distinct, non-aligned centers; two primitives on every shell.
        // --profile isolates this matrix for before/after stack sampling.
        if (mode == "--profile" || mode == "--sample-zero" ||
            mode == "--sample-finite") {
            const bool sampling = mode != "--profile";
            if (sampling && mode != "--sample-" + field_name) {
                continue;
            }
            for (std::uint16_t angular = 0; angular <= 2; ++angular) {
                if (sampling && angular != 2U) {
                    continue;
                }
                const giao::Shell a({-0.3, 0.2, 0.5}, angular,
                                     {2.1, 0.55}, {0.3, 0.8});
                const giao::Shell b({0.6, -0.4, 0.1}, angular,
                                     {1.7, 0.42}, {-0.2, 0.9});
                const giao::Shell c({0.2, 0.7, -0.6}, angular,
                                     {2.1, 0.55}, {0.3, 0.8});
                const giao::Shell d({-0.5, -0.1, 0.4}, angular,
                                     {1.7, 0.42}, {-0.2, 0.9});
                const std::array<std::string, 3> classes{
                    "profile-s-s-s-s", "profile-p-p-p-p", "profile-d-d-d-d"};
                run_eri_case(a, b, c, d, classes[angular], "2x2x2x2",
                             field, field_name, sampling ? 10.0 : minimum_seconds);
            }
            continue;
        }
        run_pair_cases(s_a, p_b, "s-p", "2x2", field, field_name,
                       minimum_seconds);
        run_pair_cases(d_a, d_b, "d-d", "3x3", field, field_name,
                       minimum_seconds);
        run_eri_case(s_a, s_b, s_a, s_b, "s-s-s-s", "2x2x2x2", field,
                     field_name, minimum_seconds);
        run_eri_case(p_a, p_b, p_a, p_b, "p-p-p-p", "2x2x2x2", field,
                     field_name, minimum_seconds);
    }
}

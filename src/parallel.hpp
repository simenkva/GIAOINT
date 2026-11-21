#pragma once

#include <atomic>
#include <cstddef>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>

namespace giao::detail {

// Each task owns its output region; primitive summation order stays serial.
template <typename Function>
void parallel_for(std::size_t count, std::size_t threads, Function&& function) {
    if (threads == 0U || threads > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("threads must be a positive integer <= 2147483647");
    }
#ifndef GIAO_HAS_OPENMP
    if (threads > 1U) {
        throw std::invalid_argument("parallel evaluation requires an OpenMP-enabled build");
    }
#endif
    if (threads == 1U || count == 0U) {
        for (std::size_t index = 0; index < count; ++index) {
            function(index);
        }
        return;
    }
#ifdef GIAO_HAS_OPENMP
    std::exception_ptr failure;
    std::mutex failure_mutex;
    std::atomic<bool> failed{false};
    const int requested = static_cast<int>(threads);
#pragma omp parallel for num_threads(requested) schedule(dynamic, 1)
    for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(count); ++index) {
        if (!failed.load(std::memory_order_relaxed)) {
            try {
                function(static_cast<std::size_t>(index));
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
                std::scoped_lock lock(failure_mutex);
                if (failure == nullptr) {
                    failure = std::current_exception();
                }
            }
        }
    }
    if (failure != nullptr) {
        std::rethrow_exception(failure);
    }
#endif
}

} // namespace giao::detail

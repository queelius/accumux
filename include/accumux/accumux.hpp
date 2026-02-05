/**
 * @file accumux.hpp
 * @brief Main header - includes all accumux components
 *
 * This is the convenience header for the accumux library.
 * Include this file to get access to all accumulators and composition tools.
 *
 * @version 2.0.0
 * @copyright MIT License
 */

#pragma once

// Core components
#include "core/accumulator_concept.hpp"
#include "core/composition.hpp"
#include "core/variadic_composition.hpp"
#include "core/algebra.hpp"

// Basic accumulators
#include "accumulators/basic.hpp"
#include "accumulators/kbn_sum.hpp"
#include "accumulators/welford.hpp"

// Extended accumulators
#include "accumulators/ema.hpp"
#include "accumulators/covariance.hpp"
#include "accumulators/histogram.hpp"
#include "accumulators/quantile.hpp"

// Performance features
#include "core/simd_support.hpp"

// Distributed computing
#include "core/thread_safe.hpp"
#include "core/serialization.hpp"
#include "core/distributed.hpp"

/**
 * @namespace accumux
 * @brief Main namespace for the accumux library
 *
 * Contains all accumulators, composition tools, and utilities for
 * algebraic online data reduction.
 *
 * ## Quick Start
 *
 * @code{.cpp}
 * #include <accumux/accumux.hpp>
 * using namespace accumux;
 *
 * // Compose multiple accumulators
 * auto stats = kbn_sum<double>() + welford_accumulator<double>() + minmax_accumulator<double>();
 *
 * // Process data
 * for (double value : data) {
 *     stats += value;
 * }
 *
 * // Extract results
 * auto [sum, variance, range] = stats.eval();
 * @endcode
 *
 * ## Available Accumulators
 *
 * ### Basic Accumulators
 * - kbn_sum<T>: Numerically stable summation (KBN algorithm)
 * - welford_accumulator<T>: Online mean and variance
 * - min_accumulator<T>, max_accumulator<T>: Extrema tracking
 * - minmax_accumulator<T>: Combined min/max
 * - count_accumulator: Element counting
 * - product_accumulator<T>: Product with overflow protection
 *
 * ### Extended Accumulators
 * - ema_accumulator<T>: Exponential moving average
 * - covariance_accumulator<T>: Online covariance and correlation
 * - histogram_accumulator<T>: Fixed-bin histogram
 * - p2_quantile_accumulator<T>: P² quantile estimation
 * - reservoir_quantile_accumulator<T>: Reservoir-based quantiles
 *
 * ## Composition Operators
 *
 * - a + b: Parallel composition (both process same data)
 * - a * b: Sequential composition (pipeline)
 * - conditional(a, b, pred): Predicate-based switching
 * - make_parallel(a, b, c, ...): Variadic parallel composition
 *
 * ## Thread Safety
 *
 * - mutex_accumulator<A>: Simple mutex-based thread safety
 * - rw_accumulator<A>: Read-write lock for read-heavy workloads
 * - sharded_accumulator<A>: Sharded for high contention
 *
 * ## Distributed Computing
 *
 * - map_reduce_accumulator<A>: Parallel processing with merge
 * - windowed_accumulator<A>: Time-based sliding window
 * - sliding_window_accumulator<A>: Count-based sliding window
 * - Serialization support for network transmission
 */

namespace accumux {

/**
 * @brief Library version information
 */
struct version {
    static constexpr int major = 2;
    static constexpr int minor = 0;
    static constexpr int patch = 0;
    static constexpr const char* string = "2.0.0";
};

} // namespace accumux

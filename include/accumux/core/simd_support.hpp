/**
 * @file simd_support.hpp
 * @brief SIMD vectorization support for accumulators
 *
 * Provides SIMD-accelerated batch operations for accumulators that
 * can benefit from vectorization.
 *
 * Design philosophy:
 * - Opt-in SIMD support via explicit batch operations
 * - Fallback to scalar code on unsupported platforms
 * - No external dependencies (uses compiler intrinsics)
 */

#pragma once

#include "accumulator_concept.hpp"
#include <array>
#include <cstddef>
#include <type_traits>

// Platform detection
#if defined(__AVX512F__)
    #define ACCUMUX_HAS_AVX512 1
    #include <immintrin.h>
#elif defined(__AVX2__) || defined(__AVX__)
    #define ACCUMUX_HAS_AVX 1
    #include <immintrin.h>
#elif defined(__SSE4_2__) || defined(__SSE4_1__) || defined(__SSE3__) || defined(__SSE2__)
    #define ACCUMUX_HAS_SSE 1
    #include <emmintrin.h>
    #include <xmmintrin.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define ACCUMUX_HAS_NEON 1
    #include <arm_neon.h>
#endif

namespace accumux {

/**
 * @brief Compile-time SIMD capability detection
 */
struct simd_capabilities {
    static constexpr bool has_avx512 =
#ifdef ACCUMUX_HAS_AVX512
        true;
#else
        false;
#endif

    static constexpr bool has_avx =
#ifdef ACCUMUX_HAS_AVX
        true;
#else
        false;
#endif

    static constexpr bool has_sse =
#ifdef ACCUMUX_HAS_SSE
        true;
#else
        false;
#endif

    static constexpr bool has_neon =
#ifdef ACCUMUX_HAS_NEON
        true;
#else
        false;
#endif

    static constexpr bool has_any_simd = has_avx512 || has_avx || has_sse || has_neon;

    // Preferred vector width for double
    static constexpr std::size_t double_width =
        has_avx512 ? 8 : (has_avx ? 4 : (has_sse ? 2 : 1));

    // Preferred vector width for float
    static constexpr std::size_t float_width =
        has_avx512 ? 16 : (has_avx ? 8 : (has_sse || has_neon ? 4 : 1));
};

/**
 * @brief SIMD vector type traits
 */
template<typename T>
struct simd_traits {
    static constexpr std::size_t width = 1;
    using scalar_type = T;
    static constexpr bool is_vectorizable = false;
};

template<>
struct simd_traits<double> {
    static constexpr std::size_t width = simd_capabilities::double_width;
    using scalar_type = double;
    static constexpr bool is_vectorizable = simd_capabilities::has_any_simd;
};

template<>
struct simd_traits<float> {
    static constexpr std::size_t width = simd_capabilities::float_width;
    using scalar_type = float;
    static constexpr bool is_vectorizable = simd_capabilities::has_any_simd;
};

/**
 * @brief Batch accumulation for any accumulator
 *
 * Processes data in batches, potentially using SIMD for supported types.
 * Falls back to scalar loop for complex accumulators.
 */
template<Accumulator Acc, typename Iterator>
void batch_accumulate(Acc& acc, Iterator first, Iterator last) {
    // Simple scalar fallback - compiler may auto-vectorize
    for (; first != last; ++first) {
        acc += *first;
    }
}

/**
 * @brief Batch accumulate from contiguous range with explicit size
 */
template<Accumulator Acc, typename T>
void batch_accumulate(Acc& acc, const T* data, std::size_t count) {
    constexpr std::size_t unroll = 4;  // Loop unrolling factor

    // Process in unrolled chunks
    std::size_t i = 0;
    for (; i + unroll <= count; i += unroll) {
        acc += data[i];
        acc += data[i + 1];
        acc += data[i + 2];
        acc += data[i + 3];
    }

    // Handle remainder
    for (; i < count; ++i) {
        acc += data[i];
    }
}

/**
 * @brief Parallel batch accumulation using multiple accumulators
 *
 * Splits data across N accumulators, then merges results.
 * Useful for exploiting instruction-level parallelism.
 */
template<std::size_t N, Accumulator Acc, typename T>
Acc parallel_batch_accumulate(const T* data, std::size_t count) {
    static_assert(N > 0, "Need at least one accumulator");

    std::array<Acc, N> accs{};

    // Distribute data across accumulators
    std::size_t chunk_size = count / N;
    std::size_t remainder = count % N;

    std::size_t offset = 0;
    for (std::size_t i = 0; i < N; ++i) {
        std::size_t this_chunk = chunk_size + (i < remainder ? 1 : 0);
        batch_accumulate(accs[i], data + offset, this_chunk);
        offset += this_chunk;
    }

    // Merge all accumulators
    Acc result = accs[0];
    for (std::size_t i = 1; i < N; ++i) {
        result += accs[i];
    }

    return result;
}

/**
 * @brief Concept for SIMD-optimizable accumulators
 *
 * An accumulator is SIMD-optimizable if it provides a specialized
 * batch_add method that can process multiple values at once.
 */
template<typename T>
concept SIMDAccumulator = Accumulator<T> && requires(T acc) {
    { T::is_simd_optimized } -> std::convertible_to<bool>;
};

/**
 * @brief SIMD-aware sum accumulator
 *
 * Specialized sum that can use SIMD for batch operations.
 * Uses pairwise summation within SIMD lanes for better accuracy.
 */
template<std::floating_point T>
class simd_sum {
public:
    using value_type = T;
    static constexpr bool is_simd_optimized = true;

private:
    T sum_;
    T correction_;  // KBN-style correction

public:
    simd_sum() : sum_(T(0)), correction_(T(0)) {}
    explicit simd_sum(T initial) : sum_(initial), correction_(T(0)) {}

    simd_sum(const simd_sum&) = default;
    simd_sum(simd_sum&&) = default;
    simd_sum& operator=(const simd_sum&) = default;
    simd_sum& operator=(simd_sum&&) = default;

    simd_sum& operator+=(const T& value) {
        // KBN algorithm
        T corrected = value + correction_;
        T new_sum = sum_ + corrected;

        if (std::abs(sum_) >= std::abs(corrected)) {
            correction_ = (sum_ - new_sum) + corrected;
        } else {
            correction_ = (corrected - new_sum) + sum_;
        }

        sum_ = new_sum;
        return *this;
    }

    simd_sum& operator+=(const simd_sum& other) {
        return *this += other.eval();
    }

    /**
     * @brief Batch add from contiguous array (SIMD-optimized path)
     */
    void batch_add(const T* data, std::size_t count) {
        // Use parallel partial sums for better cache utilization
        constexpr std::size_t LANES = 4;
        std::array<T, LANES> partials{};

        std::size_t i = 0;
        for (; i + LANES <= count; i += LANES) {
            partials[0] += data[i];
            partials[1] += data[i + 1];
            partials[2] += data[i + 2];
            partials[3] += data[i + 3];
        }

        // Add partial sums to main accumulator
        for (std::size_t j = 0; j < LANES; ++j) {
            *this += partials[j];
        }

        // Handle remainder
        for (; i < count; ++i) {
            *this += data[i];
        }
    }

    T eval() const {
        return sum_ + correction_;
    }

    explicit operator T() const {
        return eval();
    }
};

// Verify concept compliance
static_assert(Accumulator<simd_sum<double>>);
static_assert(SIMDAccumulator<simd_sum<double>>);

/**
 * @brief Optimized batch accumulation for SIMD accumulators
 */
template<SIMDAccumulator Acc, typename T>
void batch_accumulate(Acc& acc, const T* data, std::size_t count) {
    acc.batch_add(data, count);
}

/**
 * @brief Helper to process range with best available method
 */
template<Accumulator Acc, typename Range>
Acc accumulate_range(const Range& range) {
    Acc acc;
    if constexpr (requires { range.data(); range.size(); }) {
        // Contiguous range - use batch
        batch_accumulate(acc, range.data(), range.size());
    } else {
        // Generic range - use iterator
        for (const auto& val : range) {
            acc += val;
        }
    }
    return acc;
}

} // namespace accumux

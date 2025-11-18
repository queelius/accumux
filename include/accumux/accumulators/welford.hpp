/**
 * @file welford.hpp
 * @brief Welford's online algorithm for mean and variance
 * 
 * Implements Welford's algorithm for computing running mean and variance
 * in a single pass with numerical stability.
 */

#pragma once

#include "../core/accumulator_concept.hpp"
#include "kbn_sum.hpp"
#include <cmath>

namespace accumux {

/**
 * @brief Welford accumulator for mean and variance calculation
 * 
 * Implements Welford's online algorithm for computing sample statistics
 * in a single pass. Uses KBN summation internally for maximum numerical
 * stability.
 * 
 * Mathematical properties:
 * - Computes sample mean incrementally
 * - Computes both population and sample variance
 * - Numerically stable for large datasets
 * - O(1) space complexity
 * 
 * @tparam T Floating-point value type
 */
template<std::floating_point T>
class welford_accumulator {
public:
    using value_type = T;
    
private:
    std::size_t count_;         ///< Number of samples processed
    kbn_sum<T> mean_;          ///< Running mean (using KBN sum)
    kbn_sum<T> m2_;            ///< Sum of squared differences from mean
    
public:
    /**
     * @brief Default constructor - empty accumulator
     */
    welford_accumulator() : count_(0), mean_(), m2_() {}
    
    /**
     * @brief Value constructor - initialize with first sample
     */
    explicit welford_accumulator(const T& initial_value) : welford_accumulator() {
        *this += initial_value;
    }
    
    /**
     * @brief Copy constructor
     */
    welford_accumulator(const welford_accumulator&) = default;
    
    /**
     * @brief Move constructor
     */
    welford_accumulator(welford_accumulator&&) = default;
    
    /**
     * @brief Copy assignment
     */
    welford_accumulator& operator=(const welford_accumulator&) = default;
    
    /**
     * @brief Move assignment
     */
    welford_accumulator& operator=(welford_accumulator&&) = default;
    
    /**
     * @brief Add a sample using Welford's algorithm
     */
    welford_accumulator& operator+=(const T& value) {
        ++count_;
        const T delta = value - mean_.eval();
        mean_ += delta / static_cast<T>(count_);
        const T delta2 = value - mean_.eval();
        m2_ += delta * delta2;
        return *this;
    }
    
    /**
     * @brief Combine with another Welford accumulator
     * 
     * Uses the parallel algorithm for combining Welford accumulators
     * while maintaining numerical stability.
     */
    welford_accumulator& operator+=(const welford_accumulator& other) {
        if (other.count_ == 0) return *this;
        if (count_ == 0) {
            *this = other;
            return *this;
        }
        
        const auto new_count = count_ + other.count_;
        const auto delta = other.mean_.eval() - mean_.eval();
        
        // Update mean using weighted combination
        mean_ = (static_cast<T>(count_) * mean_.eval() + 
                static_cast<T>(other.count_) * other.mean_.eval()) / static_cast<T>(new_count);
        
        // Update M2 using parallel combination formula
        m2_ += other.m2_ + delta * delta * static_cast<T>(count_) * 
               static_cast<T>(other.count_) / static_cast<T>(new_count);
        
        count_ = new_count;
        return *this;
    }
    
    /**
     * @brief Binary addition operator
     */
    welford_accumulator operator+(const welford_accumulator& other) const {
        welford_accumulator result = *this;
        result += other;
        return result;
    }
    
    /**
     * @brief Get sample mean (primary result)
     */
    T eval() const {
        return mean();
    }
    
    /**
     * @brief Conversion operator returns mean
     */
    explicit operator T() const {
        return mean();
    }
    
    // Statistical interface
    
    /**
     * @brief Get sample mean
     */
    T mean() const {
        return count_ > 0 ? mean_.eval() : T(0);
    }
    
    /**
     * @brief Get population variance (divide by n)
     */
    T variance() const {
        return count_ > 0 ? m2_.eval() / static_cast<T>(count_) : T(0);
    }
    
    /**
     * @brief Get sample variance (divide by n-1)
     */
    T sample_variance() const {
        return count_ > 1 ? m2_.eval() / static_cast<T>(count_ - 1) : T(0);
    }
    
    /**
     * @brief Get standard deviation
     */
    T std_dev() const {
        return std::sqrt(variance());
    }
    
    /**
     * @brief Get sample standard deviation
     */
    T sample_std_dev() const {
        return std::sqrt(sample_variance());
    }
    
    /**
     * @brief Get total sum (mean * count)
     */
    T sum() const {
        return mean_.eval() * static_cast<T>(count_);
    }
    
    /**
     * @brief Get number of samples
     */
    std::size_t size() const {
        return count_;
    }
    
    /**
     * @brief Check if accumulator is empty
     */
    bool empty() const {
        return count_ == 0;
    }
    
    /**
     * @brief Get sum of squared deviations (for advanced use)
     */
    T sum_of_squares() const {
        return m2_.eval();
    }
};

// Static assertions to verify concept compliance
static_assert(Accumulator<welford_accumulator<double>>);
static_assert(StatisticalAccumulator<welford_accumulator<double>>);
static_assert(VarianceAccumulator<welford_accumulator<double>>);

/**
 * @brief Factory function for type deduction
 */
template<typename T>
auto make_welford_accumulator(const T& initial_value) {
    return welford_accumulator<T>(initial_value);
}

/**
 * @brief Factory function for empty accumulator
 */
template<typename T = double>
auto make_welford_accumulator() {
    return welford_accumulator<T>();
}

// Global utility functions for convenience

/**
 * @brief Compute mean of a sequence using Welford accumulator
 */
template<typename Iterator>
auto mean(Iterator first, Iterator last) {
    using value_type = std::decay_t<decltype(*first)>;
    welford_accumulator<value_type> acc;
    for (auto it = first; it != last; ++it) {
        acc += *it;
    }
    return acc.mean();
}

/**
 * @brief Compute variance of a sequence using Welford accumulator
 */
template<typename Iterator>
auto variance(Iterator first, Iterator last) {
    using value_type = std::decay_t<decltype(*first)>;
    welford_accumulator<value_type> acc;
    for (auto it = first; it != last; ++it) {
        acc += *it;
    }
    return acc.variance();
}

} // namespace accumux
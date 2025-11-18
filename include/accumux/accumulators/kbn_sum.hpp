/**
 * @file kbn_sum.hpp
 * @brief Kahan-Babuška-Neumaier summation accumulator
 * 
 * Implements numerically stable summation using the KBN algorithm
 * to minimize floating-point rounding errors.
 */

#pragma once

#include "../core/accumulator_concept.hpp"
#include <type_traits>

namespace accumux {

/**
 * @brief Kahan-Babuška-Neumaier sum accumulator
 * 
 * A numerically stable accumulator for computing sums of floating-point
 * numbers with minimal rounding error. Uses compensated summation to
 * maintain accuracy even with large sequences of values.
 * 
 * Mathematical properties:
 * - Monoid over (kbn_sum, +, kbn_sum())
 * - Homomorphisms to/from (T, +, T(0))
 * - Error bound: O(1) vs O(n) for naive summation
 * 
 * @tparam T Floating-point value type
 */
template<std::floating_point T>
class kbn_sum {
public:
    using value_type = T;
    
private:
    T sum_;         ///< Running sum
    T correction_;  ///< Correction term for numerical errors
    
public:
    /**
     * @brief Default constructor - creates zero accumulator
     */
    kbn_sum() : sum_(T(0)), correction_(T(0)) {}
    
    /**
     * @brief Value constructor - initialize with single value
     */
    explicit kbn_sum(const T& initial_value) : sum_(initial_value), correction_(T(0)) {}
    
    /**
     * @brief Copy constructor
     */
    kbn_sum(const kbn_sum&) = default;
    
    /**
     * @brief Move constructor
     */
    kbn_sum(kbn_sum&&) = default;
    
    /**
     * @brief Copy assignment
     */
    kbn_sum& operator=(const kbn_sum&) = default;
    
    /**
     * @brief Move assignment
     */
    kbn_sum& operator=(kbn_sum&&) = default;
    
    /**
     * @brief Value assignment - reset to single value
     */
    kbn_sum& operator=(const T& value) {
        sum_ = value;
        correction_ = T(0);
        return *this;
    }
    
    /**
     * @brief Add a value using KBN algorithm
     * 
     * Implements the compensated summation algorithm to minimize
     * numerical errors in floating-point addition.
     */
    kbn_sum& operator+=(const T& value) {
        // Kahan-Babuška-Neumaier algorithm
        const T corrected_next_term = value + correction_;
        const T new_sum = sum_ + corrected_next_term;
        
        if (std::abs(sum_) >= std::abs(corrected_next_term)) {
            correction_ = (sum_ - new_sum) + corrected_next_term;
        } else {
            correction_ = (corrected_next_term - new_sum) + sum_;
        }
        
        sum_ = new_sum;
        return *this;
    }
    
    /**
     * @brief Combine with another kbn_sum accumulator
     */
    kbn_sum& operator+=(const kbn_sum& other) {
        // Add the other's total (sum + correction) to this accumulator
        return *this += other.eval();
    }
    
    /**
     * @brief Binary addition operator
     */
    kbn_sum operator+(const kbn_sum& other) const {
        kbn_sum result = *this;
        result += other;
        return result;
    }
    
    /**
     * @brief Binary addition with value
     */
    kbn_sum operator+(const T& value) const {
        kbn_sum result = *this;
        result += value;
        return result;
    }
    
    /**
     * @brief Get the current sum (primary interface)
     */
    T eval() const {
        return sum_ + correction_;
    }
    
    /**
     * @brief Conversion operator to value type
     */
    explicit operator T() const {
        return eval();
    }
    
    /**
     * @brief Comparison operators
     */
    bool operator==(const kbn_sum& other) const {
        return eval() == other.eval();
    }
    
    bool operator<(const kbn_sum& other) const {
        return eval() < other.eval();
    }
    
    bool operator<(const T& value) const {
        return eval() < value;
    }
    
    /**
     * @brief Absolute value function
     */
    kbn_sum abs() const {
        return kbn_sum(std::abs(eval()));
    }
    
    /**
     * @brief Access to internal components (for advanced use)
     */
    T sum_component() const { return sum_; }
    T correction_component() const { return correction_; }
};

// Static assertions to verify concept compliance
static_assert(Accumulator<kbn_sum<double>>);
static_assert(std::is_same_v<typename kbn_sum<float>::value_type, float>);

/**
 * @brief Factory function for type deduction
 */
template<typename T>
auto make_kbn_sum(const T& initial_value = T(0)) {
    return kbn_sum<T>(initial_value);
}

/**
 * @brief Absolute value function for kbn_sum
 */
template<std::floating_point T>
kbn_sum<T> abs(const kbn_sum<T>& accumulator) {
    return accumulator.abs();
}

} // namespace accumux
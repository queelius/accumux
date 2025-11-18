/**
 * @file basic.hpp
 * @brief Basic accumulator types (min, max, count, product)
 * 
 * Simple accumulator types for common reduction operations.
 */

#pragma once

#include "../core/accumulator_concept.hpp"
#include <algorithm>
#include <limits>
#include <cstddef>
#include <cmath>

namespace accumux {

/**
 * @brief Minimum value accumulator
 * 
 * Tracks the minimum value seen in a stream of data.
 * 
 * @tparam T Comparable value type
 */
template<typename T>
class min_accumulator {
public:
    using value_type = T;
    
private:
    T min_value_;
    bool has_value_;
    
public:
    /**
     * @brief Default constructor - no minimum yet
     */
    min_accumulator() : min_value_(std::numeric_limits<T>::max()), has_value_(false) {}
    
    /**
     * @brief Value constructor - initialize with first value
     */
    explicit min_accumulator(const T& initial_value) : min_value_(initial_value), has_value_(true) {}
    
    /**
     * @brief Standard copy/move constructors and assignments
     */
    min_accumulator(const min_accumulator&) = default;
    min_accumulator(min_accumulator&&) = default;
    min_accumulator& operator=(const min_accumulator&) = default;
    min_accumulator& operator=(min_accumulator&&) = default;
    
    /**
     * @brief Process a new value
     */
    min_accumulator& operator+=(const T& value) {
        if (!has_value_ || value < min_value_) {
            min_value_ = value;
            has_value_ = true;
        }
        return *this;
    }
    
    /**
     * @brief Combine with another min accumulator
     */
    min_accumulator& operator+=(const min_accumulator& other) {
        if (other.has_value_) {
            *this += other.min_value_;
        }
        return *this;
    }
    
    /**
     * @brief Get minimum value
     */
    T eval() const {
        return has_value_ ? min_value_ : std::numeric_limits<T>::max();
    }
    
    /**
     * @brief Conversion operator
     */
    explicit operator T() const {
        return eval();
    }
    
    /**
     * @brief Check if we have seen any values
     */
    bool empty() const {
        return !has_value_;
    }
};

/**
 * @brief Maximum value accumulator
 * 
 * Tracks the maximum value seen in a stream of data.
 * 
 * @tparam T Comparable value type
 */
template<typename T>
class max_accumulator {
public:
    using value_type = T;
    
private:
    T max_value_;
    bool has_value_;
    
public:
    max_accumulator() : max_value_(std::numeric_limits<T>::lowest()), has_value_(false) {}
    explicit max_accumulator(const T& initial_value) : max_value_(initial_value), has_value_(true) {}
    
    max_accumulator(const max_accumulator&) = default;
    max_accumulator(max_accumulator&&) = default;
    max_accumulator& operator=(const max_accumulator&) = default;
    max_accumulator& operator=(max_accumulator&&) = default;
    
    max_accumulator& operator+=(const T& value) {
        if (!has_value_ || value > max_value_) {
            max_value_ = value;
            has_value_ = true;
        }
        return *this;
    }
    
    max_accumulator& operator+=(const max_accumulator& other) {
        if (other.has_value_) {
            *this += other.max_value_;
        }
        return *this;
    }
    
    T eval() const {
        return has_value_ ? max_value_ : std::numeric_limits<T>::lowest();
    }
    
    explicit operator T() const {
        return eval();
    }
    
    bool empty() const {
        return !has_value_;
    }
};

/**
 * @brief Count accumulator
 * 
 * Simply counts the number of items processed.
 */
class count_accumulator {
public:
    using value_type = std::size_t;
    
private:
    std::size_t count_;
    
public:
    count_accumulator() : count_(0) {}
    explicit count_accumulator(std::size_t initial_count) : count_(initial_count) {}
    
    count_accumulator(const count_accumulator&) = default;
    count_accumulator(count_accumulator&&) = default;
    count_accumulator& operator=(const count_accumulator&) = default;
    count_accumulator& operator=(count_accumulator&&) = default;
    
    template<typename T>
    count_accumulator& operator+=(const T&) {
        ++count_;
        return *this;
    }
    
    count_accumulator& operator+=(const count_accumulator& other) {
        count_ += other.count_;
        return *this;
    }
    
    std::size_t eval() const {
        return count_;
    }
    
    explicit operator std::size_t() const {
        return count_;
    }
    
    std::size_t size() const {
        return count_;
    }
};

/**
 * @brief Product accumulator
 * 
 * Computes the product of all values. Uses logarithmic representation
 * internally to avoid overflow/underflow for large sequences.
 * 
 * @tparam T Numeric value type
 */
template<std::floating_point T>
class product_accumulator {
public:
    using value_type = T;
    
private:
    T log_product_;  ///< Sum of logarithms
    bool has_value_;
    bool has_zero_;  ///< Track if we've seen zero
    
public:
    product_accumulator() : log_product_(T(0)), has_value_(false), has_zero_(false) {}
    
    explicit product_accumulator(const T& initial_value) : product_accumulator() {
        *this += initial_value;
    }
    
    product_accumulator(const product_accumulator&) = default;
    product_accumulator(product_accumulator&&) = default;
    product_accumulator& operator=(const product_accumulator&) = default;
    product_accumulator& operator=(product_accumulator&&) = default;
    
    product_accumulator& operator+=(const T& value) {
        if (value == T(0)) {
            has_zero_ = true;
        } else {
            log_product_ += std::log(std::abs(value));
            has_value_ = true;
        }
        return *this;
    }
    
    product_accumulator& operator+=(const product_accumulator& other) {
        if (other.has_zero_) {
            has_zero_ = true;
        }
        if (other.has_value_) {
            log_product_ += other.log_product_;
            has_value_ = true;
        }
        return *this;
    }
    
    T eval() const {
        if (has_zero_) return T(0);
        if (!has_value_) return T(1);
        return std::exp(log_product_);
    }
    
    explicit operator T() const {
        return eval();
    }
    
    bool empty() const {
        return !has_value_ && !has_zero_;
    }
};

// Static assertions to verify concept compliance
static_assert(Accumulator<min_accumulator<double>>);
static_assert(Accumulator<max_accumulator<int>>);
static_assert(Accumulator<count_accumulator>);
static_assert(Accumulator<product_accumulator<double>>);

/**
 * @brief Min-Max accumulator that tracks both simultaneously
 * 
 * More efficient than composing separate min and max accumulators.
 */
template<typename T>
class minmax_accumulator {
public:
    using value_type = std::pair<T, T>;
    
private:
    T min_value_, max_value_;
    bool has_value_;
    
public:
    minmax_accumulator() : 
        min_value_(std::numeric_limits<T>::max()),
        max_value_(std::numeric_limits<T>::lowest()),
        has_value_(false) {}
    
    explicit minmax_accumulator(const T& initial_value) : 
        min_value_(initial_value), max_value_(initial_value), has_value_(true) {}
    
    minmax_accumulator(const std::pair<T, T>& initial_pair) :
        min_value_(initial_pair.first), max_value_(initial_pair.second), has_value_(true) {}
    
    minmax_accumulator(const minmax_accumulator&) = default;
    minmax_accumulator(minmax_accumulator&&) = default;
    minmax_accumulator& operator=(const minmax_accumulator&) = default;
    minmax_accumulator& operator=(minmax_accumulator&&) = default;
    
    minmax_accumulator& operator+=(const T& value) {
        if (!has_value_) {
            min_value_ = max_value_ = value;
            has_value_ = true;
        } else {
            min_value_ = std::min(min_value_, value);
            max_value_ = std::max(max_value_, value);
        }
        return *this;
    }
    
    minmax_accumulator& operator+=(const minmax_accumulator& other) {
        if (other.has_value_) {
            *this += other.min_value_;
            *this += other.max_value_;
        }
        return *this;
    }
    
    std::pair<T, T> eval() const {
        return has_value_ ? std::make_pair(min_value_, max_value_) : 
                          std::make_pair(std::numeric_limits<T>::max(), std::numeric_limits<T>::lowest());
    }
    
    explicit operator std::pair<T, T>() const {
        return eval();
    }
    
    T min() const { return min_value_; }
    T max() const { return max_value_; }
    T range() const { return max_value_ - min_value_; }
    bool empty() const { return !has_value_; }
};

static_assert(Accumulator<minmax_accumulator<double>>);

// Factory functions
template<typename T>
auto make_min_accumulator(const T& initial = T{}) {
    return min_accumulator<T>(initial);
}

template<typename T>
auto make_max_accumulator(const T& initial = T{}) {
    return max_accumulator<T>(initial);
}

template<typename T>
auto make_minmax_accumulator(const T& initial = T{}) {
    return minmax_accumulator<T>(initial);
}

inline auto make_count_accumulator() {
    return count_accumulator();
}

template<typename T>
auto make_product_accumulator(const T& initial = T(1)) {
    return product_accumulator<T>(initial);
}

} // namespace accumux
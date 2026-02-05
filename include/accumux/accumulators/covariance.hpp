/**
 * @file covariance.hpp
 * @brief Online covariance and correlation accumulator
 *
 * Implements numerically stable online covariance computation using
 * an extension of Welford's algorithm to bivariate data.
 */

#pragma once

#include "../core/accumulator_concept.hpp"
#include "kbn_sum.hpp"
#include <cmath>
#include <utility>
#include <tuple>

namespace accumux {

/**
 * @brief Online covariance accumulator for bivariate data
 *
 * Computes covariance, correlation, and individual statistics for
 * paired (x, y) observations using a numerically stable online algorithm.
 *
 * Based on the parallel algorithm by Chan et al. (1983) extended to
 * covariance computation.
 *
 * Mathematical properties:
 * - Single-pass computation of covariance
 * - Numerically stable via compensated arithmetic
 * - O(1) space complexity
 * - Supports combining partial results
 *
 * @tparam T Floating-point value type
 */
template<std::floating_point T>
class covariance_accumulator {
public:
    using value_type = T;  // Returns covariance
    using input_type = std::pair<T, T>;

private:
    std::size_t count_;
    kbn_sum<T> mean_x_;
    kbn_sum<T> mean_y_;
    kbn_sum<T> m2_x_;      ///< Sum of squared deviations for x
    kbn_sum<T> m2_y_;      ///< Sum of squared deviations for y
    kbn_sum<T> c_xy_;      ///< Sum of (x - mean_x)(y - mean_y)

public:
    covariance_accumulator()
        : count_(0), mean_x_(), mean_y_(), m2_x_(), m2_y_(), c_xy_() {}

    covariance_accumulator(const covariance_accumulator&) = default;
    covariance_accumulator(covariance_accumulator&&) = default;
    covariance_accumulator& operator=(const covariance_accumulator&) = default;
    covariance_accumulator& operator=(covariance_accumulator&&) = default;

    /**
     * @brief Add a (x, y) pair using online algorithm
     */
    covariance_accumulator& operator+=(const std::pair<T, T>& xy) {
        ++count_;
        T n = static_cast<T>(count_);

        T dx = xy.first - mean_x_.eval();
        T dy = xy.second - mean_y_.eval();

        mean_x_ += dx / n;
        mean_y_ += dy / n;

        T dx2 = xy.first - mean_x_.eval();
        T dy2 = xy.second - mean_y_.eval();

        m2_x_ += dx * dx2;
        m2_y_ += dy * dy2;
        c_xy_ += dx * dy2;  // Use updated mean for y

        return *this;
    }

    /**
     * @brief Add a single value (treats as (value, value) - for concept compliance)
     */
    covariance_accumulator& operator+=(const T& value) {
        return *this += std::make_pair(value, value);
    }

    /**
     * @brief Combine with another covariance accumulator (parallel algorithm)
     */
    covariance_accumulator& operator+=(const covariance_accumulator& other) {
        if (other.count_ == 0) return *this;
        if (count_ == 0) {
            *this = other;
            return *this;
        }

        T n1 = static_cast<T>(count_);
        T n2 = static_cast<T>(other.count_);
        T n = n1 + n2;

        T dx = other.mean_x_.eval() - mean_x_.eval();
        T dy = other.mean_y_.eval() - mean_y_.eval();

        // Update means
        mean_x_ = (n1 * mean_x_.eval() + n2 * other.mean_x_.eval()) / n;
        mean_y_ = (n1 * mean_y_.eval() + n2 * other.mean_y_.eval()) / n;

        // Update M2 values
        m2_x_ += other.m2_x_ + dx * dx * n1 * n2 / n;
        m2_y_ += other.m2_y_ + dy * dy * n1 * n2 / n;

        // Update covariance term
        c_xy_ += other.c_xy_ + dx * dy * n1 * n2 / n;

        count_ += other.count_;
        return *this;
    }

    /**
     * @brief Get sample covariance (primary result)
     */
    T eval() const {
        return sample_covariance();
    }

    explicit operator T() const {
        return eval();
    }

    // Statistical interface

    /**
     * @brief Number of observations
     */
    std::size_t size() const { return count_; }

    /**
     * @brief Mean of x values
     */
    T mean_x() const {
        return count_ > 0 ? mean_x_.eval() : T(0);
    }

    /**
     * @brief Mean of y values
     */
    T mean_y() const {
        return count_ > 0 ? mean_y_.eval() : T(0);
    }

    /**
     * @brief Mean (returns mean_x for concept compliance)
     */
    T mean() const { return mean_x(); }

    /**
     * @brief Population covariance (divide by n)
     */
    T covariance() const {
        return count_ > 0 ? c_xy_.eval() / static_cast<T>(count_) : T(0);
    }

    /**
     * @brief Sample covariance (divide by n-1)
     */
    T sample_covariance() const {
        return count_ > 1 ? c_xy_.eval() / static_cast<T>(count_ - 1) : T(0);
    }

    /**
     * @brief Population variance of x
     */
    T variance_x() const {
        return count_ > 0 ? m2_x_.eval() / static_cast<T>(count_) : T(0);
    }

    /**
     * @brief Population variance of y
     */
    T variance_y() const {
        return count_ > 0 ? m2_y_.eval() / static_cast<T>(count_) : T(0);
    }

    /**
     * @brief Sample variance of x
     */
    T sample_variance_x() const {
        return count_ > 1 ? m2_x_.eval() / static_cast<T>(count_ - 1) : T(0);
    }

    /**
     * @brief Sample variance of y
     */
    T sample_variance_y() const {
        return count_ > 1 ? m2_y_.eval() / static_cast<T>(count_ - 1) : T(0);
    }

    /**
     * @brief Standard deviation of x
     */
    T std_dev_x() const { return std::sqrt(variance_x()); }

    /**
     * @brief Standard deviation of y
     */
    T std_dev_y() const { return std::sqrt(variance_y()); }

    /**
     * @brief Pearson correlation coefficient
     */
    T correlation() const {
        if (count_ < 2) return T(0);
        T sx = std_dev_x();
        T sy = std_dev_y();
        if (sx == T(0) || sy == T(0)) return T(0);
        return covariance() / (sx * sy);
    }

    /**
     * @brief Slope of linear regression (y = a + b*x)
     */
    T slope() const {
        T vx = variance_x();
        return vx > T(0) ? covariance() / vx : T(0);
    }

    /**
     * @brief Intercept of linear regression (y = a + b*x)
     */
    T intercept() const {
        return mean_y() - slope() * mean_x();
    }

    /**
     * @brief R-squared (coefficient of determination)
     */
    T r_squared() const {
        T r = correlation();
        return r * r;
    }

    /**
     * @brief Check if empty
     */
    bool empty() const { return count_ == 0; }
};

// Verify concept compliance
static_assert(Accumulator<covariance_accumulator<double>>);

/**
 * @brief Factory function
 */
template<typename T = double>
auto make_covariance_accumulator() {
    return covariance_accumulator<T>();
}

/**
 * @brief Convenience function to compute correlation of ranges
 */
template<typename IterX, typename IterY>
auto correlation(IterX first_x, IterX last_x, IterY first_y) {
    using T = std::decay_t<decltype(*first_x)>;
    covariance_accumulator<T> acc;
    for (; first_x != last_x; ++first_x, ++first_y) {
        acc += std::make_pair(*first_x, *first_y);
    }
    return acc.correlation();
}

} // namespace accumux

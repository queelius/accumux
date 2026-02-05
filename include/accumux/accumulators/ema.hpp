/**
 * @file ema.hpp
 * @brief Exponential Moving Average accumulator
 *
 * Implements numerically stable exponential moving average (EMA) for
 * streaming time series data with configurable smoothing factor.
 */

#pragma once

#include "../core/accumulator_concept.hpp"
#include <cmath>
#include <stdexcept>

namespace accumux {

/**
 * @brief Exponential Moving Average accumulator
 *
 * Computes exponential moving average with configurable smoothing factor alpha.
 * EMA_t = alpha * x_t + (1 - alpha) * EMA_{t-1}
 *
 * For time series with period N, common choice is alpha = 2/(N+1)
 *
 * Mathematical properties:
 * - Weights decay exponentially: w_k = alpha * (1-alpha)^k
 * - More recent values have higher weight
 * - O(1) space and time per update
 *
 * @tparam T Floating-point value type
 */
template<std::floating_point T>
class ema_accumulator {
public:
    using value_type = T;

private:
    T alpha_;           ///< Smoothing factor (0 < alpha <= 1)
    T ema_;             ///< Current EMA value
    T ema_variance_;    ///< EMA of squared differences (for volatility)
    std::size_t count_; ///< Number of samples processed
    bool initialized_;  ///< Whether we have at least one value

public:
    /**
     * @brief Construct EMA with smoothing factor
     * @param alpha Smoothing factor (0 < alpha <= 1), default 0.1
     * @throws std::invalid_argument if alpha not in (0, 1]
     */
    explicit ema_accumulator(T alpha = T(0.1))
        : alpha_(alpha), ema_(T(0)), ema_variance_(T(0)), count_(0), initialized_(false) {
        if (alpha <= T(0) || alpha > T(1)) {
            throw std::invalid_argument("EMA alpha must be in range (0, 1]");
        }
    }

    /**
     * @brief Construct EMA from period (alpha = 2/(N+1))
     */
    static ema_accumulator from_period(std::size_t period) {
        if (period == 0) {
            throw std::invalid_argument("EMA period must be > 0");
        }
        return ema_accumulator(T(2) / (T(period) + T(1)));
    }

    /**
     * @brief Construct EMA from half-life (alpha = 1 - exp(-ln(2)/half_life))
     */
    static ema_accumulator from_half_life(T half_life) {
        if (half_life <= T(0)) {
            throw std::invalid_argument("EMA half-life must be > 0");
        }
        return ema_accumulator(T(1) - std::exp(-std::log(T(2)) / half_life));
    }

    ema_accumulator(const ema_accumulator&) = default;
    ema_accumulator(ema_accumulator&&) = default;
    ema_accumulator& operator=(const ema_accumulator&) = default;
    ema_accumulator& operator=(ema_accumulator&&) = default;

    /**
     * @brief Add a value to the EMA
     */
    ema_accumulator& operator+=(const T& value) {
        ++count_;
        if (!initialized_) {
            ema_ = value;
            ema_variance_ = T(0);
            initialized_ = true;
        } else {
            T delta = value - ema_;
            ema_ += alpha_ * delta;
            // Update EMA variance (for volatility tracking)
            ema_variance_ = (T(1) - alpha_) * (ema_variance_ + alpha_ * delta * delta);
        }
        return *this;
    }

    /**
     * @brief Combine two EMA accumulators
     *
     * Note: Combining EMAs is approximate - we weight by count
     */
    ema_accumulator& operator+=(const ema_accumulator& other) {
        if (!other.initialized_) return *this;
        if (!initialized_) {
            *this = other;
            return *this;
        }

        // Weighted combination based on effective sample sizes
        T total = static_cast<T>(count_ + other.count_);
        T w1 = static_cast<T>(count_) / total;
        T w2 = static_cast<T>(other.count_) / total;

        ema_ = w1 * ema_ + w2 * other.ema_;
        ema_variance_ = w1 * ema_variance_ + w2 * other.ema_variance_;
        count_ += other.count_;

        return *this;
    }

    /**
     * @brief Get current EMA value
     */
    T eval() const {
        return ema_;
    }

    explicit operator T() const {
        return eval();
    }

    // Extended interface

    /**
     * @brief Get smoothing factor
     */
    T alpha() const { return alpha_; }

    /**
     * @brief Get number of samples
     */
    std::size_t size() const { return count_; }

    /**
     * @brief Get EMA mean (same as eval)
     */
    T mean() const { return ema_; }

    /**
     * @brief Get EMA variance (volatility measure)
     */
    T variance() const { return ema_variance_; }

    /**
     * @brief Get EMA standard deviation
     */
    T std_dev() const { return std::sqrt(ema_variance_); }

    /**
     * @brief Check if accumulator is empty
     */
    bool empty() const { return !initialized_; }

    /**
     * @brief Get effective number of samples (1/alpha for infinite series)
     */
    T effective_samples() const {
        return T(1) / alpha_;
    }
};

// Verify concept compliance
static_assert(Accumulator<ema_accumulator<double>>);
static_assert(StatisticalAccumulator<ema_accumulator<double>>);

/**
 * @brief Factory functions
 */
template<typename T = double>
auto make_ema_accumulator(T alpha = T(0.1)) {
    return ema_accumulator<T>(alpha);
}

template<typename T = double>
auto make_ema_from_period(std::size_t period) {
    return ema_accumulator<T>::from_period(period);
}

template<typename T = double>
auto make_ema_from_half_life(T half_life) {
    return ema_accumulator<T>::from_half_life(half_life);
}

} // namespace accumux

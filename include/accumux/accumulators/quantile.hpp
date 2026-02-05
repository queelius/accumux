/**
 * @file quantile.hpp
 * @brief Online quantile estimation accumulators
 *
 * Implements streaming quantile estimation using the P² algorithm
 * (Jain & Chlamtac, 1985) and a simple reservoir-based approach.
 */

#pragma once

#include "../core/accumulator_concept.hpp"
#include <array>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace accumux {

/**
 * @brief P² quantile accumulator (Jain & Chlamtac algorithm)
 *
 * Estimates a single quantile using O(1) space with the P² algorithm.
 * Maintains 5 markers that converge to the desired quantile.
 *
 * Mathematical properties:
 * - O(1) space complexity
 * - O(1) update time
 * - Approximate (not exact) quantile estimation
 * - Accuracy improves with more data
 *
 * Reference: Jain & Chlamtac, "The P² Algorithm for Dynamic Calculation
 * of Quantiles and Histograms Without Storing Observations", 1985.
 *
 * @tparam T Floating-point value type
 */
template<std::floating_point T>
class p2_quantile_accumulator {
public:
    using value_type = T;

private:
    T p_;                         ///< Target quantile (0-1)
    std::array<T, 5> q_;          ///< Marker heights (quantile estimates)
    std::array<int, 5> n_;        ///< Marker positions
    std::array<T, 5> n_prime_;    ///< Desired marker positions
    std::array<T, 5> dn_;         ///< Increments for n_prime
    std::size_t count_;           ///< Number of observations

    /**
     * @brief Piecewise parabolic interpolation (P²)
     */
    T parabolic(int i, int d) const {
        T qi = q_[i];
        T qim1 = q_[i - 1];
        T qip1 = q_[i + 1];
        int ni = n_[i];
        int nim1 = n_[i - 1];
        int nip1 = n_[i + 1];

        T num = static_cast<T>(d) / static_cast<T>(nip1 - nim1) *
                ((static_cast<T>(ni - nim1 + d) * (qip1 - qi) / static_cast<T>(nip1 - ni)) +
                 (static_cast<T>(nip1 - ni - d) * (qi - qim1) / static_cast<T>(ni - nim1)));

        return qi + num;
    }

    /**
     * @brief Linear interpolation fallback
     */
    T linear(int i, int d) const {
        int j = i + d;
        return q_[i] + static_cast<T>(d) * (q_[j] - q_[i]) /
               static_cast<T>(n_[j] - n_[i]);
    }

public:
    /**
     * @brief Construct P² quantile estimator
     * @param p Target quantile (0 < p < 1)
     */
    explicit p2_quantile_accumulator(T p = T(0.5))
        : p_(p), q_{}, n_{0, 1, 2, 3, 4}, n_prime_{}, dn_{}, count_(0) {
        if (p <= T(0) || p >= T(1)) {
            throw std::invalid_argument("Quantile p must be in (0, 1)");
        }

        // Initialize desired positions and increments
        n_prime_ = {T(0), T(2) * p_, T(4) * p_, T(2) + T(2) * p_, T(4)};
        dn_ = {T(0), p_ / T(2), p_, (T(1) + p_) / T(2), T(1)};
    }

    p2_quantile_accumulator(const p2_quantile_accumulator&) = default;
    p2_quantile_accumulator(p2_quantile_accumulator&&) = default;
    p2_quantile_accumulator& operator=(const p2_quantile_accumulator&) = default;
    p2_quantile_accumulator& operator=(p2_quantile_accumulator&&) = default;

    /**
     * @brief Add observation
     */
    p2_quantile_accumulator& operator+=(const T& x) {
        ++count_;

        if (count_ <= 5) {
            // Initialization phase: store first 5 observations
            q_[count_ - 1] = x;
            if (count_ == 5) {
                std::sort(q_.begin(), q_.end());
            }
            return *this;
        }

        // Find cell k where x belongs
        int k;
        if (x < q_[0]) {
            q_[0] = x;
            k = 0;
        } else if (x < q_[1]) {
            k = 0;
        } else if (x < q_[2]) {
            k = 1;
        } else if (x < q_[3]) {
            k = 2;
        } else if (x < q_[4]) {
            k = 3;
        } else {
            q_[4] = x;
            k = 3;
        }

        // Increment positions of markers k+1 through 4
        for (int i = k + 1; i < 5; ++i) {
            ++n_[i];
        }

        // Update desired positions
        for (int i = 0; i < 5; ++i) {
            n_prime_[i] += dn_[i];
        }

        // Adjust marker heights 1-3 if necessary
        for (int i = 1; i < 4; ++i) {
            T d = n_prime_[i] - static_cast<T>(n_[i]);

            if ((d >= T(1) && n_[i + 1] - n_[i] > 1) ||
                (d <= T(-1) && n_[i - 1] - n_[i] < -1)) {
                int di = (d >= T(0)) ? 1 : -1;

                // Try parabolic formula
                T q_new = parabolic(i, di);

                // Use linear if parabolic is out of bounds
                if (q_new <= q_[i - 1] || q_new >= q_[i + 1]) {
                    q_new = linear(i, di);
                }

                q_[i] = q_new;
                n_[i] += di;
            }
        }

        return *this;
    }

    /**
     * @brief Combine with another P² accumulator (approximate)
     */
    p2_quantile_accumulator& operator+=(const p2_quantile_accumulator& other) {
        if (other.count_ == 0) return *this;
        if (count_ < 5) {
            // In initialization phase, just add values
            for (std::size_t i = 0; i < other.count_ && i < 5; ++i) {
                *this += other.q_[i];
            }
            return *this;
        }

        // Weighted combination of estimates (approximate)
        T w1 = static_cast<T>(count_) / static_cast<T>(count_ + other.count_);
        T w2 = T(1) - w1;

        for (int i = 0; i < 5; ++i) {
            q_[i] = w1 * q_[i] + w2 * other.q_[i];
        }
        count_ += other.count_;

        return *this;
    }

    /**
     * @brief Get quantile estimate
     */
    T eval() const {
        if (count_ < 5) {
            // Not enough data - return median of available
            std::array<T, 5> sorted = q_;
            std::sort(sorted.begin(), sorted.begin() + count_);
            return sorted[count_ / 2];
        }
        return q_[2];  // Middle marker is the quantile estimate
    }

    explicit operator T() const { return eval(); }

    // Extended interface
    T target_quantile() const { return p_; }
    std::size_t size() const { return count_; }
    T mean() const { return eval(); }  // For concept compliance
    bool empty() const { return count_ == 0; }

    /**
     * @brief Get all marker values (for diagnostics)
     */
    std::array<T, 5> markers() const { return q_; }
};

// Verify concept compliance
static_assert(Accumulator<p2_quantile_accumulator<double>>);

/**
 * @brief Reservoir-based exact quantile accumulator
 *
 * Maintains a random sample of observations for exact quantile computation.
 * Uses reservoir sampling for bounded memory with uniform sampling.
 *
 * @tparam T Value type
 */
template<typename T>
class reservoir_quantile_accumulator {
public:
    using value_type = T;

private:
    std::vector<T> reservoir_;
    std::size_t max_size_;
    std::size_t count_;
    mutable std::mt19937 rng_;

public:
    /**
     * @brief Construct with reservoir size
     * @param max_size Maximum samples to store
     * @param seed Random seed (default: random)
     */
    explicit reservoir_quantile_accumulator(std::size_t max_size = 10000,
                                            unsigned int seed = std::random_device{}())
        : reservoir_(), max_size_(max_size), count_(0), rng_(seed) {
        reservoir_.reserve(max_size_);
    }

    reservoir_quantile_accumulator(const reservoir_quantile_accumulator&) = default;
    reservoir_quantile_accumulator(reservoir_quantile_accumulator&&) = default;
    reservoir_quantile_accumulator& operator=(const reservoir_quantile_accumulator&) = default;
    reservoir_quantile_accumulator& operator=(reservoir_quantile_accumulator&&) = default;

    /**
     * @brief Add observation using reservoir sampling
     */
    reservoir_quantile_accumulator& operator+=(const T& value) {
        ++count_;

        if (reservoir_.size() < max_size_) {
            reservoir_.push_back(value);
        } else {
            // Reservoir sampling: replace with probability max_size_/count_
            std::uniform_int_distribution<std::size_t> dist(0, count_ - 1);
            std::size_t j = dist(rng_);
            if (j < max_size_) {
                reservoir_[j] = value;
            }
        }
        return *this;
    }

    /**
     * @brief Combine reservoirs
     */
    reservoir_quantile_accumulator& operator+=(const reservoir_quantile_accumulator& other) {
        for (const auto& val : other.reservoir_) {
            *this += val;
        }
        return *this;
    }

    /**
     * @brief Get median (default quantile)
     */
    T eval() const {
        return quantile(0.5);
    }

    explicit operator T() const { return eval(); }

    /**
     * @brief Get arbitrary quantile
     */
    T quantile(double p) const {
        if (reservoir_.empty()) return T{};
        if (p <= 0.0) return *std::min_element(reservoir_.begin(), reservoir_.end());
        if (p >= 1.0) return *std::max_element(reservoir_.begin(), reservoir_.end());

        std::vector<T> sorted = reservoir_;
        std::sort(sorted.begin(), sorted.end());

        double idx = p * (sorted.size() - 1);
        std::size_t lo = static_cast<std::size_t>(idx);
        std::size_t hi = std::min(lo + 1, sorted.size() - 1);
        double frac = idx - static_cast<double>(lo);

        return static_cast<T>(sorted[lo] * (1.0 - frac) + sorted[hi] * frac);
    }

    /**
     * @brief Get multiple quantiles efficiently
     */
    std::vector<T> quantiles(const std::vector<double>& ps) const {
        if (reservoir_.empty()) return std::vector<T>(ps.size(), T{});

        std::vector<T> sorted = reservoir_;
        std::sort(sorted.begin(), sorted.end());

        std::vector<T> result;
        result.reserve(ps.size());

        for (double p : ps) {
            if (p <= 0.0) {
                result.push_back(sorted.front());
            } else if (p >= 1.0) {
                result.push_back(sorted.back());
            } else {
                double idx = p * (sorted.size() - 1);
                std::size_t lo = static_cast<std::size_t>(idx);
                std::size_t hi = std::min(lo + 1, sorted.size() - 1);
                double frac = idx - static_cast<double>(lo);
                result.push_back(static_cast<T>(sorted[lo] * (1.0 - frac) + sorted[hi] * frac));
            }
        }
        return result;
    }

    T median() const { return quantile(0.5); }
    T q1() const { return quantile(0.25); }
    T q3() const { return quantile(0.75); }
    T iqr() const { return q3() - q1(); }

    T mean() const {
        if (reservoir_.empty()) return T{};
        T sum = T{};
        for (const auto& v : reservoir_) sum += v;
        return sum / static_cast<T>(reservoir_.size());
    }

    std::size_t size() const { return count_; }
    std::size_t reservoir_size() const { return reservoir_.size(); }
    std::size_t max_reservoir_size() const { return max_size_; }
    bool empty() const { return count_ == 0; }
};

// Verify concept compliance
static_assert(Accumulator<reservoir_quantile_accumulator<double>>);

/**
 * @brief Factory functions
 */
template<typename T = double>
auto make_p2_quantile(T p = T(0.5)) {
    return p2_quantile_accumulator<T>(p);
}

template<typename T = double>
auto make_median_accumulator() {
    return p2_quantile_accumulator<T>(T(0.5));
}

template<typename T = double>
auto make_reservoir_quantile(std::size_t max_size = 10000) {
    return reservoir_quantile_accumulator<T>(max_size);
}

} // namespace accumux

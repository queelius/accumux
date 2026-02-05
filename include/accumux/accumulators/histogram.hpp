/**
 * @file histogram.hpp
 * @brief Online histogram accumulator with fixed and adaptive binning
 *
 * Implements streaming histogram computation with O(B) space where B is
 * the number of bins.
 */

#pragma once

#include "../core/accumulator_concept.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <numeric>
#include <limits>

namespace accumux {

/**
 * @brief Fixed-bin histogram accumulator
 *
 * Maintains a histogram with fixed bin boundaries. Values outside
 * the range are counted in underflow/overflow bins.
 *
 * Mathematical properties:
 * - O(B) space for B bins
 * - O(1) update time
 * - Exact counts for fixed ranges
 * - Supports combining partial histograms
 *
 * @tparam T Numeric value type
 */
template<typename T>
class histogram_accumulator {
public:
    using value_type = T;  // Input value type (bin counts available via counts())

private:
    T min_;                           ///< Minimum bin boundary
    T max_;                           ///< Maximum bin boundary
    std::size_t num_bins_;            ///< Number of bins
    T bin_width_;                     ///< Width of each bin
    std::vector<std::size_t> counts_; ///< Bin counts
    std::size_t underflow_;           ///< Count below min
    std::size_t overflow_;            ///< Count above max
    std::size_t total_;               ///< Total count

public:
    /**
     * @brief Construct histogram with fixed bins
     * @param min Minimum value (left edge of first bin)
     * @param max Maximum value (right edge of last bin)
     * @param num_bins Number of bins
     */
    histogram_accumulator(T min, T max, std::size_t num_bins)
        : min_(min), max_(max), num_bins_(num_bins),
          bin_width_((max - min) / static_cast<T>(num_bins)),
          counts_(num_bins, 0), underflow_(0), overflow_(0), total_(0) {
        if (min >= max) {
            throw std::invalid_argument("Histogram min must be less than max");
        }
        if (num_bins == 0) {
            throw std::invalid_argument("Histogram must have at least 1 bin");
        }
    }

    /**
     * @brief Default constructor (1000 bins from 0 to 1)
     */
    histogram_accumulator() : histogram_accumulator(T(0), T(1), 100) {}

    histogram_accumulator(const histogram_accumulator&) = default;
    histogram_accumulator(histogram_accumulator&&) = default;
    histogram_accumulator& operator=(const histogram_accumulator&) = default;
    histogram_accumulator& operator=(histogram_accumulator&&) = default;

    /**
     * @brief Add a value to the histogram
     */
    histogram_accumulator& operator+=(const T& value) {
        ++total_;
        if (value < min_) {
            ++underflow_;
        } else if (value >= max_) {
            ++overflow_;
        } else {
            std::size_t bin = static_cast<std::size_t>((value - min_) / bin_width_);
            bin = std::min(bin, num_bins_ - 1);  // Handle edge case
            ++counts_[bin];
        }
        return *this;
    }

    /**
     * @brief Combine with another histogram (must have same bins)
     */
    histogram_accumulator& operator+=(const histogram_accumulator& other) {
        if (min_ != other.min_ || max_ != other.max_ || num_bins_ != other.num_bins_) {
            throw std::invalid_argument("Cannot combine histograms with different bins");
        }
        for (std::size_t i = 0; i < num_bins_; ++i) {
            counts_[i] += other.counts_[i];
        }
        underflow_ += other.underflow_;
        overflow_ += other.overflow_;
        total_ += other.total_;
        return *this;
    }

    /**
     * @brief Get estimated mean from histogram
     */
    T eval() const {
        return mean();
    }

    explicit operator T() const {
        return eval();
    }

    // Extended interface

    /**
     * @brief Get count for specific bin
     */
    std::size_t bin_count(std::size_t bin) const {
        return bin < num_bins_ ? counts_[bin] : 0;
    }

    /**
     * @brief Get left edge of a bin
     */
    T bin_left(std::size_t bin) const {
        return min_ + static_cast<T>(bin) * bin_width_;
    }

    /**
     * @brief Get right edge of a bin
     */
    T bin_right(std::size_t bin) const {
        return min_ + static_cast<T>(bin + 1) * bin_width_;
    }

    /**
     * @brief Get center of a bin
     */
    T bin_center(std::size_t bin) const {
        return min_ + (static_cast<T>(bin) + T(0.5)) * bin_width_;
    }

    /**
     * @brief Get bin index for a value
     */
    std::size_t bin_for(const T& value) const {
        if (value < min_) return std::numeric_limits<std::size_t>::max();
        if (value >= max_) return std::numeric_limits<std::size_t>::max();
        return static_cast<std::size_t>((value - min_) / bin_width_);
    }

    /**
     * @brief Get probability density at bin (count / (total * bin_width))
     */
    double density(std::size_t bin) const {
        if (total_ == 0 || bin >= num_bins_) return 0.0;
        return static_cast<double>(counts_[bin]) /
               (static_cast<double>(total_) * static_cast<double>(bin_width_));
    }

    /**
     * @brief Get relative frequency of bin (count / total)
     */
    double frequency(std::size_t bin) const {
        if (total_ == 0 || bin >= num_bins_) return 0.0;
        return static_cast<double>(counts_[bin]) / static_cast<double>(total_);
    }

    /**
     * @brief Get cumulative count up to and including bin
     */
    std::size_t cumulative_count(std::size_t bin) const {
        std::size_t sum = underflow_;
        for (std::size_t i = 0; i <= bin && i < num_bins_; ++i) {
            sum += counts_[i];
        }
        return sum;
    }

    /**
     * @brief Get cumulative distribution function value at bin
     */
    double cdf(std::size_t bin) const {
        return total_ > 0 ? static_cast<double>(cumulative_count(bin)) / static_cast<double>(total_) : 0.0;
    }

    /**
     * @brief Estimate quantile (linear interpolation)
     */
    T quantile(double p) const {
        if (total_ == 0 || p < 0.0 || p > 1.0) return min_;

        std::size_t target = static_cast<std::size_t>(p * static_cast<double>(total_));
        std::size_t cumsum = underflow_;

        for (std::size_t i = 0; i < num_bins_; ++i) {
            if (cumsum + counts_[i] >= target) {
                // Linear interpolation within bin
                double frac = counts_[i] > 0 ?
                    static_cast<double>(target - cumsum) / static_cast<double>(counts_[i]) : 0.0;
                return bin_left(i) + static_cast<T>(frac) * bin_width_;
            }
            cumsum += counts_[i];
        }
        return max_;
    }

    /**
     * @brief Estimate median
     */
    T median() const { return quantile(0.5); }

    /**
     * @brief Estimate mean from histogram
     */
    T mean() const {
        if (total_ == 0) return T(0);
        T sum = T(0);
        for (std::size_t i = 0; i < num_bins_; ++i) {
            sum += bin_center(i) * static_cast<T>(counts_[i]);
        }
        return sum / static_cast<T>(total_ - underflow_ - overflow_);
    }

    // Accessors
    T min() const { return min_; }
    T max() const { return max_; }
    std::size_t num_bins() const { return num_bins_; }
    T bin_width() const { return bin_width_; }
    std::size_t underflow() const { return underflow_; }
    std::size_t overflow() const { return overflow_; }
    std::size_t total() const { return total_; }
    std::size_t size() const { return total_; }
    bool empty() const { return total_ == 0; }

    const std::vector<std::size_t>& counts() const { return counts_; }
};

// Note: histogram_accumulator has value_type = vector<size_t>, which requires
// special handling. The basic Accumulator concept check is relaxed here.

/**
 * @brief Factory function
 */
template<typename T = double>
auto make_histogram(T min, T max, std::size_t num_bins) {
    return histogram_accumulator<T>(min, max, num_bins);
}

/**
 * @brief Convenience function: create histogram and populate from range
 */
template<typename Iterator>
auto histogram(Iterator first, Iterator last, std::size_t num_bins = 100) {
    using T = std::decay_t<decltype(*first)>;

    // First pass: find min/max
    T min_val = std::numeric_limits<T>::max();
    T max_val = std::numeric_limits<T>::lowest();
    for (auto it = first; it != last; ++it) {
        min_val = std::min(min_val, *it);
        max_val = std::max(max_val, *it);
    }

    // Add small padding to include max value
    T padding = (max_val - min_val) * T(0.001);
    if (padding == T(0)) padding = T(1);

    histogram_accumulator<T> hist(min_val, max_val + padding, num_bins);

    // Second pass: fill histogram
    for (auto it = first; it != last; ++it) {
        hist += *it;
    }

    return hist;
}

} // namespace accumux

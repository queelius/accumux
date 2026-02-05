/**
 * @file distributed.hpp
 * @brief Distributed accumulation patterns
 *
 * Provides patterns for distributed computing including:
 * - Map-reduce style accumulation
 * - Hierarchical merging
 * - Windowed accumulation for streaming
 */

#pragma once

#include "accumulator_concept.hpp"
#include "thread_safe.hpp"
#include <vector>
#include <functional>
#include <future>
#include <queue>
#include <chrono>
#include <optional>
#include <deque>

namespace accumux {

/**
 * @brief Map-reduce accumulator pattern
 *
 * Distributes data processing across multiple threads, then merges results.
 *
 * @tparam Acc Accumulator type
 * @tparam Mapper Function type for mapping input to accumulator values
 */
template<Accumulator Acc>
class map_reduce_accumulator {
public:
    using value_type = typename Acc::value_type;

private:
    std::size_t num_workers_;

public:
    explicit map_reduce_accumulator(std::size_t num_workers = 0)
        : num_workers_(num_workers == 0 ? std::thread::hardware_concurrency() : num_workers) {
        if (num_workers_ == 0) num_workers_ = 4;
    }

    /**
     * @brief Process range in parallel and return merged accumulator
     */
    template<typename Iterator>
    Acc process(Iterator first, Iterator last) const {
        auto total = std::distance(first, last);
        if (total == 0) return Acc{};
        if (total < static_cast<std::ptrdiff_t>(num_workers_)) {
            // Not enough data to parallelize
            Acc acc;
            for (; first != last; ++first) {
                acc += *first;
            }
            return acc;
        }

        // Launch workers
        std::vector<std::future<Acc>> futures;
        futures.reserve(num_workers_);

        auto chunk_size = total / static_cast<std::ptrdiff_t>(num_workers_);
        auto remainder = total % static_cast<std::ptrdiff_t>(num_workers_);

        Iterator chunk_start = first;
        for (std::size_t i = 0; i < num_workers_; ++i) {
            auto this_chunk = chunk_size + (static_cast<std::ptrdiff_t>(i) < remainder ? 1 : 0);
            Iterator chunk_end = chunk_start;
            std::advance(chunk_end, this_chunk);

            futures.push_back(std::async(std::launch::async,
                [](Iterator start, Iterator end) {
                    Acc local_acc;
                    for (; start != end; ++start) {
                        local_acc += *start;
                    }
                    return local_acc;
                }, chunk_start, chunk_end));

            chunk_start = chunk_end;
        }

        // Merge results
        Acc result;
        for (auto& f : futures) {
            result += f.get();
        }
        return result;
    }

    /**
     * @brief Process with custom mapper function
     */
    template<typename Iterator, typename Mapper>
    Acc process(Iterator first, Iterator last, Mapper&& mapper) const {
        auto total = std::distance(first, last);
        if (total == 0) return Acc{};

        std::vector<std::future<Acc>> futures;
        futures.reserve(num_workers_);

        auto chunk_size = total / static_cast<std::ptrdiff_t>(num_workers_);
        auto remainder = total % static_cast<std::ptrdiff_t>(num_workers_);

        Iterator chunk_start = first;
        for (std::size_t i = 0; i < num_workers_; ++i) {
            auto this_chunk = chunk_size + (static_cast<std::ptrdiff_t>(i) < remainder ? 1 : 0);
            Iterator chunk_end = chunk_start;
            std::advance(chunk_end, this_chunk);

            futures.push_back(std::async(std::launch::async,
                [&mapper](Iterator start, Iterator end) {
                    Acc local_acc;
                    for (; start != end; ++start) {
                        local_acc += mapper(*start);
                    }
                    return local_acc;
                }, chunk_start, chunk_end));

            chunk_start = chunk_end;
        }

        Acc result;
        for (auto& f : futures) {
            result += f.get();
        }
        return result;
    }

    std::size_t num_workers() const { return num_workers_; }
};

/**
 * @brief Hierarchical merge accumulator
 *
 * Merges partial accumulators in a tree structure for better
 * numerical stability and parallelism.
 */
template<Accumulator Acc>
class hierarchical_merge {
public:
    /**
     * @brief Merge a vector of accumulators hierarchically
     */
    static Acc merge(std::vector<Acc> accumulators) {
        if (accumulators.empty()) return Acc{};
        if (accumulators.size() == 1) return std::move(accumulators[0]);

        // Pairwise merge until one remains
        while (accumulators.size() > 1) {
            std::vector<Acc> next_level;
            next_level.reserve((accumulators.size() + 1) / 2);

            for (std::size_t i = 0; i + 1 < accumulators.size(); i += 2) {
                accumulators[i] += accumulators[i + 1];
                next_level.push_back(std::move(accumulators[i]));
            }

            // Handle odd element
            if (accumulators.size() % 2 == 1) {
                next_level.push_back(std::move(accumulators.back()));
            }

            accumulators = std::move(next_level);
        }

        return std::move(accumulators[0]);
    }

    /**
     * @brief Merge accumulators in parallel using tree reduction
     */
    static Acc parallel_merge(std::vector<Acc> accumulators) {
        if (accumulators.empty()) return Acc{};
        if (accumulators.size() == 1) return std::move(accumulators[0]);

        while (accumulators.size() > 1) {
            std::vector<std::future<Acc>> futures;
            std::vector<Acc> next_level;

            for (std::size_t i = 0; i + 1 < accumulators.size(); i += 2) {
                futures.push_back(std::async(std::launch::async,
                    [](Acc a, Acc b) {
                        a += b;
                        return a;
                    }, std::move(accumulators[i]), std::move(accumulators[i + 1])));
            }

            // Handle odd element
            if (accumulators.size() % 2 == 1) {
                next_level.push_back(std::move(accumulators.back()));
            }

            for (auto& f : futures) {
                next_level.push_back(f.get());
            }

            accumulators = std::move(next_level);
        }

        return std::move(accumulators[0]);
    }
};

/**
 * @brief Time-based windowed accumulator
 *
 * Maintains statistics over a sliding time window.
 * Old data is automatically expired.
 */
template<Accumulator Acc>
class windowed_accumulator {
public:
    using value_type = typename Acc::value_type;
    using clock = std::chrono::steady_clock;
    using duration = clock::duration;
    using time_point = clock::time_point;

private:
    struct TimedValue {
        time_point timestamp;
        typename Acc::value_type value;
    };

    duration window_size_;
    std::deque<TimedValue> values_;
    mutable Acc cached_acc_;
    mutable bool cache_valid_ = false;

    void expire_old() {
        auto cutoff = clock::now() - window_size_;
        while (!values_.empty() && values_.front().timestamp < cutoff) {
            values_.pop_front();
            cache_valid_ = false;
        }
    }

    void rebuild_cache() const {
        cached_acc_ = Acc{};
        for (const auto& tv : values_) {
            cached_acc_ += tv.value;
        }
        cache_valid_ = true;
    }

public:
    /**
     * @brief Construct with window size
     */
    explicit windowed_accumulator(duration window_size)
        : window_size_(window_size) {}

    /**
     * @brief Construct with window size in seconds
     */
    explicit windowed_accumulator(double seconds)
        : window_size_(std::chrono::duration_cast<duration>(
              std::chrono::duration<double>(seconds))) {}

    windowed_accumulator(const windowed_accumulator&) = default;
    windowed_accumulator(windowed_accumulator&&) = default;
    windowed_accumulator& operator=(const windowed_accumulator&) = default;
    windowed_accumulator& operator=(windowed_accumulator&&) = default;

    /**
     * @brief Add value with current timestamp
     */
    template<typename T>
    windowed_accumulator& operator+=(const T& value) {
        expire_old();
        values_.push_back({clock::now(), static_cast<typename Acc::value_type>(value)});
        cache_valid_ = false;
        return *this;
    }

    /**
     * @brief Add value with explicit timestamp
     */
    template<typename T>
    void add(const T& value, time_point timestamp) {
        values_.push_back({timestamp, static_cast<typename Acc::value_type>(value)});
        cache_valid_ = false;
        expire_old();
    }

    /**
     * @brief Combine with another windowed accumulator
     */
    windowed_accumulator& operator+=(const windowed_accumulator& other) {
        for (const auto& tv : other.values_) {
            values_.push_back(tv);
        }
        cache_valid_ = false;
        expire_old();
        return *this;
    }

    /**
     * @brief Get accumulated result for current window
     */
    value_type eval() const {
        const_cast<windowed_accumulator*>(this)->expire_old();
        if (!cache_valid_) {
            rebuild_cache();
        }
        return cached_acc_.eval();
    }

    explicit operator value_type() const {
        return eval();
    }

    /**
     * @brief Get accumulator for current window
     */
    Acc accumulator() const {
        const_cast<windowed_accumulator*>(this)->expire_old();
        if (!cache_valid_) {
            rebuild_cache();
        }
        return cached_acc_;
    }

    /**
     * @brief Number of values in current window
     */
    std::size_t size() const {
        const_cast<windowed_accumulator*>(this)->expire_old();
        return values_.size();
    }

    bool empty() const { return size() == 0; }

    duration window_size() const { return window_size_; }

    /**
     * @brief Clear all data
     */
    void clear() {
        values_.clear();
        cache_valid_ = false;
    }
};

/**
 * @brief Count-based sliding window accumulator
 */
template<Accumulator Acc>
class sliding_window_accumulator {
public:
    using value_type = typename Acc::value_type;

private:
    std::size_t window_size_;
    std::deque<typename Acc::value_type> values_;
    mutable Acc cached_acc_;
    mutable bool cache_valid_ = false;

    void rebuild_cache() const {
        cached_acc_ = Acc{};
        for (const auto& v : values_) {
            cached_acc_ += v;
        }
        cache_valid_ = true;
    }

public:
    explicit sliding_window_accumulator(std::size_t window_size)
        : window_size_(window_size) {
        if (window_size == 0) {
            throw std::invalid_argument("Window size must be > 0");
        }
    }

    sliding_window_accumulator(const sliding_window_accumulator&) = default;
    sliding_window_accumulator(sliding_window_accumulator&&) = default;
    sliding_window_accumulator& operator=(const sliding_window_accumulator&) = default;
    sliding_window_accumulator& operator=(sliding_window_accumulator&&) = default;

    template<typename T>
    sliding_window_accumulator& operator+=(const T& value) {
        values_.push_back(static_cast<typename Acc::value_type>(value));
        if (values_.size() > window_size_) {
            values_.pop_front();
        }
        cache_valid_ = false;
        return *this;
    }

    sliding_window_accumulator& operator+=(const sliding_window_accumulator& other) {
        for (const auto& v : other.values_) {
            *this += v;
        }
        return *this;
    }

    value_type eval() const {
        if (!cache_valid_) {
            rebuild_cache();
        }
        return cached_acc_.eval();
    }

    explicit operator value_type() const {
        return eval();
    }

    Acc accumulator() const {
        if (!cache_valid_) {
            rebuild_cache();
        }
        return cached_acc_;
    }

    std::size_t size() const { return values_.size(); }
    std::size_t window_size() const { return window_size_; }
    bool empty() const { return values_.empty(); }
    bool full() const { return values_.size() >= window_size_; }

    void clear() {
        values_.clear();
        cache_valid_ = false;
    }
};

/**
 * @brief Factory functions
 */
template<Accumulator Acc>
auto make_map_reduce(std::size_t num_workers = 0) {
    return map_reduce_accumulator<Acc>(num_workers);
}

template<Accumulator Acc>
auto make_windowed(std::chrono::steady_clock::duration window_size) {
    return windowed_accumulator<Acc>(window_size);
}

template<Accumulator Acc>
auto make_windowed(double seconds) {
    return windowed_accumulator<Acc>(seconds);
}

template<Accumulator Acc>
auto make_sliding_window(std::size_t window_size) {
    return sliding_window_accumulator<Acc>(window_size);
}

} // namespace accumux

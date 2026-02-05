/**
 * @file thread_safe.hpp
 * @brief Thread-safe accumulator wrappers
 *
 * Provides thread-safe wrappers for accumulators using various
 * synchronization strategies.
 */

#pragma once

#include "accumulator_concept.hpp"
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <memory>
#include <vector>
#include <thread>

namespace accumux {

/**
 * @brief Mutex-based thread-safe accumulator wrapper
 *
 * Simple thread-safe wrapper using a mutex for all operations.
 * Suitable for low-contention scenarios.
 *
 * @tparam Acc Underlying accumulator type
 */
template<Accumulator Acc>
class mutex_accumulator {
public:
    using value_type = typename Acc::value_type;

private:
    Acc accumulator_;
    mutable std::mutex mutex_;

public:
    mutex_accumulator() = default;

    explicit mutex_accumulator(Acc acc)
        : accumulator_(std::move(acc)) {}

    mutex_accumulator(const mutex_accumulator& other)
        : accumulator_([&other]() {
              std::lock_guard lock(other.mutex_);
              return other.accumulator_;
          }()) {}

    mutex_accumulator(mutex_accumulator&& other) noexcept
        : accumulator_([&other]() {
              std::lock_guard lock(other.mutex_);
              return std::move(other.accumulator_);
          }()) {}

    mutex_accumulator& operator=(const mutex_accumulator& other) {
        if (this != &other) {
            std::scoped_lock lock(mutex_, other.mutex_);
            accumulator_ = other.accumulator_;
        }
        return *this;
    }

    mutex_accumulator& operator=(mutex_accumulator&& other) noexcept {
        if (this != &other) {
            std::scoped_lock lock(mutex_, other.mutex_);
            accumulator_ = std::move(other.accumulator_);
        }
        return *this;
    }

    /**
     * @brief Thread-safe value accumulation
     */
    template<typename T>
    mutex_accumulator& operator+=(const T& value) {
        std::lock_guard lock(mutex_);
        accumulator_ += value;
        return *this;
    }

    /**
     * @brief Thread-safe accumulator combination
     */
    mutex_accumulator& operator+=(const mutex_accumulator& other) {
        if (this != &other) {
            std::scoped_lock lock(mutex_, other.mutex_);
            accumulator_ += other.accumulator_;
        }
        return *this;
    }

    /**
     * @brief Thread-safe result extraction
     */
    value_type eval() const {
        std::lock_guard lock(mutex_);
        return accumulator_.eval();
    }

    explicit operator value_type() const {
        return eval();
    }

    /**
     * @brief Get copy of underlying accumulator (thread-safe)
     */
    Acc snapshot() const {
        std::lock_guard lock(mutex_);
        return accumulator_;
    }

    /**
     * @brief Reset accumulator (thread-safe)
     */
    void reset() {
        std::lock_guard lock(mutex_);
        accumulator_ = Acc{};
    }

    /**
     * @brief Swap and reset (atomic get-and-clear)
     */
    Acc swap_and_reset() {
        std::lock_guard lock(mutex_);
        Acc result = std::move(accumulator_);
        accumulator_ = Acc{};
        return result;
    }
};

// Verify concept compliance
template<Accumulator Acc>
constexpr bool mutex_accumulator_is_accumulator = Accumulator<mutex_accumulator<Acc>>;

/**
 * @brief Read-write lock based thread-safe accumulator
 *
 * Uses shared_mutex for better read performance in read-heavy scenarios.
 * Writers get exclusive access, readers can proceed concurrently.
 *
 * @tparam Acc Underlying accumulator type
 */
template<Accumulator Acc>
class rw_accumulator {
public:
    using value_type = typename Acc::value_type;

private:
    Acc accumulator_;
    mutable std::shared_mutex mutex_;

public:
    rw_accumulator() = default;

    explicit rw_accumulator(Acc acc)
        : accumulator_(std::move(acc)) {}

    rw_accumulator(const rw_accumulator& other)
        : accumulator_([&other]() {
              std::shared_lock lock(other.mutex_);
              return other.accumulator_;
          }()) {}

    rw_accumulator& operator=(const rw_accumulator& other) {
        if (this != &other) {
            std::unique_lock lock1(mutex_, std::defer_lock);
            std::shared_lock lock2(other.mutex_, std::defer_lock);
            std::lock(lock1, lock2);
            accumulator_ = other.accumulator_;
        }
        return *this;
    }

    /**
     * @brief Thread-safe value accumulation (exclusive lock)
     */
    template<typename T>
    rw_accumulator& operator+=(const T& value) {
        std::unique_lock lock(mutex_);
        accumulator_ += value;
        return *this;
    }

    /**
     * @brief Thread-safe accumulator combination
     */
    rw_accumulator& operator+=(const rw_accumulator& other) {
        if (this != &other) {
            std::unique_lock lock1(mutex_, std::defer_lock);
            std::shared_lock lock2(other.mutex_, std::defer_lock);
            std::lock(lock1, lock2);
            accumulator_ += other.accumulator_;
        }
        return *this;
    }

    /**
     * @brief Thread-safe result extraction (shared lock)
     */
    value_type eval() const {
        std::shared_lock lock(mutex_);
        return accumulator_.eval();
    }

    explicit operator value_type() const {
        return eval();
    }

    Acc snapshot() const {
        std::shared_lock lock(mutex_);
        return accumulator_;
    }

    void reset() {
        std::unique_lock lock(mutex_);
        accumulator_ = Acc{};
    }
};

/**
 * @brief Sharded accumulator for high-contention scenarios
 *
 * Maintains multiple accumulator shards to reduce contention.
 * Values are accumulated into a shard based on thread ID.
 * Reading requires merging all shards.
 *
 * @tparam Acc Underlying accumulator type
 * @tparam NumShards Number of shards (default: hardware concurrency)
 */
template<Accumulator Acc, std::size_t NumShards = 0>
class sharded_accumulator {
public:
    using value_type = typename Acc::value_type;

private:
    struct Shard {
        Acc accumulator;
        std::mutex mutex;

        Shard() = default;
        Shard(const Shard&) = delete;
        Shard& operator=(const Shard&) = delete;
        Shard(Shard&&) = delete;
        Shard& operator=(Shard&&) = delete;
    };

    std::vector<std::unique_ptr<Shard>> shards_;
    std::size_t num_shards_;

    std::size_t get_shard_index() const {
        // Use thread ID hash for shard selection
        auto id = std::hash<std::thread::id>{}(std::this_thread::get_id());
        return id % num_shards_;
    }

public:
    explicit sharded_accumulator(std::size_t num_shards = 0)
        : num_shards_(num_shards == 0 ? std::thread::hardware_concurrency() : num_shards) {
        if (num_shards_ == 0) num_shards_ = 4;  // Fallback
        shards_.reserve(num_shards_);
        for (std::size_t i = 0; i < num_shards_; ++i) {
            shards_.push_back(std::make_unique<Shard>());
        }
    }

    // Copy constructor - creates new shards with copied accumulators
    sharded_accumulator(const sharded_accumulator& other)
        : num_shards_(other.num_shards_) {
        shards_.reserve(num_shards_);
        for (std::size_t i = 0; i < num_shards_; ++i) {
            shards_.push_back(std::make_unique<Shard>());
            std::lock_guard lock(other.shards_[i]->mutex);
            shards_[i]->accumulator = other.shards_[i]->accumulator;
        }
    }

    // Move constructor
    sharded_accumulator(sharded_accumulator&& other) noexcept
        : shards_(std::move(other.shards_)), num_shards_(other.num_shards_) {
        other.num_shards_ = 0;
    }

    sharded_accumulator& operator=(const sharded_accumulator& other) {
        if (this != &other) {
            // Create new shards
            std::vector<std::unique_ptr<Shard>> new_shards;
            new_shards.reserve(other.num_shards_);
            for (std::size_t i = 0; i < other.num_shards_; ++i) {
                new_shards.push_back(std::make_unique<Shard>());
                std::lock_guard lock(other.shards_[i]->mutex);
                new_shards[i]->accumulator = other.shards_[i]->accumulator;
            }
            shards_ = std::move(new_shards);
            num_shards_ = other.num_shards_;
        }
        return *this;
    }

    sharded_accumulator& operator=(sharded_accumulator&& other) noexcept {
        if (this != &other) {
            shards_ = std::move(other.shards_);
            num_shards_ = other.num_shards_;
            other.num_shards_ = 0;
        }
        return *this;
    }

    /**
     * @brief Add value to appropriate shard
     */
    template<typename T>
    sharded_accumulator& operator+=(const T& value) {
        std::size_t idx = get_shard_index();
        std::lock_guard lock(shards_[idx]->mutex);
        shards_[idx]->accumulator += value;
        return *this;
    }

    /**
     * @brief Combine with another sharded accumulator
     */
    sharded_accumulator& operator+=(const sharded_accumulator& other) {
        // Merge each shard from other into corresponding shard here
        for (std::size_t i = 0; i < std::min(num_shards_, other.num_shards_); ++i) {
            std::scoped_lock lock(shards_[i]->mutex, other.shards_[i]->mutex);
            shards_[i]->accumulator += other.shards_[i]->accumulator;
        }
        return *this;
    }

    /**
     * @brief Merge all shards and get result
     */
    value_type eval() const {
        Acc merged;
        for (std::size_t i = 0; i < num_shards_; ++i) {
            std::lock_guard lock(shards_[i]->mutex);
            merged += shards_[i]->accumulator;
        }
        return merged.eval();
    }

    explicit operator value_type() const {
        return eval();
    }

    /**
     * @brief Get merged snapshot
     */
    Acc snapshot() const {
        Acc merged;
        for (std::size_t i = 0; i < num_shards_; ++i) {
            std::lock_guard lock(shards_[i]->mutex);
            merged += shards_[i]->accumulator;
        }
        return merged;
    }

    /**
     * @brief Reset all shards
     */
    void reset() {
        for (std::size_t i = 0; i < num_shards_; ++i) {
            std::lock_guard lock(shards_[i]->mutex);
            shards_[i]->accumulator = Acc{};
        }
    }

    std::size_t shard_count() const { return num_shards_; }
};

/**
 * @brief Factory functions
 */
template<Accumulator Acc>
auto make_thread_safe(Acc acc) {
    return mutex_accumulator<Acc>(std::move(acc));
}

template<Accumulator Acc>
auto make_rw_safe(Acc acc) {
    return rw_accumulator<Acc>(std::move(acc));
}

template<Accumulator Acc>
auto make_sharded(std::size_t num_shards = 0) {
    return sharded_accumulator<Acc>(num_shards);
}

} // namespace accumux

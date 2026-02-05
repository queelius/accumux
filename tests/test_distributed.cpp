/**
 * @file test_distributed.cpp
 * @brief Tests for distributed computing patterns
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <thread>
#include <chrono>
#include <numeric>

#include "../include/accumux/core/thread_safe.hpp"
#include "../include/accumux/core/distributed.hpp"
#include "../include/accumux/core/serialization.hpp"
#include "../include/accumux/accumulators/kbn_sum.hpp"
#include "../include/accumux/accumulators/welford.hpp"
#include "../include/accumux/accumulators/basic.hpp"

using namespace accumux;

// ============================================================================
// Thread Safety Tests
// ============================================================================

class ThreadSafetyTest : public ::testing::Test {
protected:
    static constexpr double EPSILON = 1e-10;
};

TEST_F(ThreadSafetyTest, MutexAccumulatorBasic) {
    mutex_accumulator<kbn_sum<double>> safe_sum;

    safe_sum += 1.0;
    safe_sum += 2.0;
    safe_sum += 3.0;

    EXPECT_NEAR(safe_sum.eval(), 6.0, EPSILON);
}

TEST_F(ThreadSafetyTest, MutexAccumulatorMultiThread) {
    mutex_accumulator<kbn_sum<double>> safe_sum;

    constexpr int NUM_THREADS = 4;
    constexpr int VALUES_PER_THREAD = 1000;

    std::vector<std::thread> threads;
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&safe_sum]() {
            for (int i = 0; i < VALUES_PER_THREAD; ++i) {
                safe_sum += 1.0;
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_NEAR(safe_sum.eval(), NUM_THREADS * VALUES_PER_THREAD, EPSILON);
}

TEST_F(ThreadSafetyTest, MutexAccumulatorSnapshot) {
    mutex_accumulator<welford_accumulator<double>> safe_welford;

    for (int i = 1; i <= 100; ++i) {
        safe_welford += static_cast<double>(i);
    }

    auto snapshot = safe_welford.snapshot();
    EXPECT_EQ(snapshot.size(), 100);
    EXPECT_NEAR(snapshot.mean(), 50.5, EPSILON);
}

TEST_F(ThreadSafetyTest, MutexAccumulatorSwapAndReset) {
    mutex_accumulator<count_accumulator> safe_count;

    safe_count += 1;
    safe_count += 1;
    safe_count += 1;

    auto old = safe_count.swap_and_reset();
    EXPECT_EQ(old.eval(), 3);
    EXPECT_EQ(safe_count.eval(), 0);
}

TEST_F(ThreadSafetyTest, ShardedAccumulatorBasic) {
    sharded_accumulator<kbn_sum<double>> sharded(4);

    sharded += 1.0;
    sharded += 2.0;
    sharded += 3.0;

    EXPECT_NEAR(sharded.eval(), 6.0, EPSILON);
}

TEST_F(ThreadSafetyTest, ShardedAccumulatorMultiThread) {
    sharded_accumulator<kbn_sum<double>> sharded;

    constexpr int NUM_THREADS = 8;
    constexpr int VALUES_PER_THREAD = 10000;

    std::vector<std::thread> threads;
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&sharded]() {
            for (int i = 0; i < VALUES_PER_THREAD; ++i) {
                sharded += 1.0;
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_NEAR(sharded.eval(), NUM_THREADS * VALUES_PER_THREAD, EPSILON);
}

TEST_F(ThreadSafetyTest, RWAccumulatorConcurrentRead) {
    rw_accumulator<welford_accumulator<double>> rw;

    for (int i = 1; i <= 1000; ++i) {
        rw += static_cast<double>(i);
    }

    // Multiple concurrent readers
    std::vector<std::thread> readers;
    std::atomic<int> read_count{0};

    for (int r = 0; r < 10; ++r) {
        readers.emplace_back([&rw, &read_count]() {
            for (int i = 0; i < 100; ++i) {
                auto val = rw.eval();
                EXPECT_NEAR(val, 500.5, 1.0);
                ++read_count;
            }
        });
    }

    for (auto& t : readers) {
        t.join();
    }

    EXPECT_EQ(read_count, 1000);
}

// ============================================================================
// Distributed Pattern Tests
// ============================================================================

class DistributedTest : public ::testing::Test {
protected:
    static constexpr double EPSILON = 1e-10;
};

TEST_F(DistributedTest, MapReduceBasic) {
    map_reduce_accumulator<kbn_sum<double>> mr(4);

    std::vector<double> data(10000);
    std::iota(data.begin(), data.end(), 1.0);  // 1, 2, ..., 10000

    auto result = mr.process(data.begin(), data.end());

    // Sum of 1 to 10000 = 10000 * 10001 / 2 = 50005000
    EXPECT_NEAR(result.eval(), 50005000.0, EPSILON);
}

TEST_F(DistributedTest, MapReduceWithMapper) {
    map_reduce_accumulator<kbn_sum<double>> mr(4);

    std::vector<int> data(100);
    std::iota(data.begin(), data.end(), 1);  // 1, 2, ..., 100

    // Sum of squares
    auto result = mr.process(data.begin(), data.end(),
        [](int x) { return static_cast<double>(x * x); });

    // Sum of squares 1 to 100 = 100*101*201/6 = 338350
    EXPECT_NEAR(result.eval(), 338350.0, EPSILON);
}

TEST_F(DistributedTest, HierarchicalMerge) {
    std::vector<kbn_sum<double>> accums(8);

    for (int i = 0; i < 8; ++i) {
        accums[i] += static_cast<double>(i + 1);
    }

    auto result = hierarchical_merge<kbn_sum<double>>::merge(std::move(accums));

    // 1 + 2 + ... + 8 = 36
    EXPECT_NEAR(result.eval(), 36.0, EPSILON);
}

TEST_F(DistributedTest, SlidingWindowBasic) {
    sliding_window_accumulator<kbn_sum<double>> window(5);

    window += 1.0;
    window += 2.0;
    window += 3.0;
    window += 4.0;
    window += 5.0;

    EXPECT_NEAR(window.eval(), 15.0, EPSILON);
    EXPECT_EQ(window.size(), 5);
    EXPECT_TRUE(window.full());

    // Adding 6th element should push out 1
    window += 6.0;
    EXPECT_NEAR(window.eval(), 20.0, EPSILON);  // 2+3+4+5+6
    EXPECT_EQ(window.size(), 5);
}

TEST_F(DistributedTest, SlidingWindowWelford) {
    sliding_window_accumulator<welford_accumulator<double>> window(10);

    for (int i = 1; i <= 20; ++i) {
        window += static_cast<double>(i);
    }

    // Window contains 11-20
    auto acc = window.accumulator();
    EXPECT_EQ(acc.size(), 10);
    EXPECT_NEAR(acc.mean(), 15.5, EPSILON);
}

// ============================================================================
// Serialization Tests
// ============================================================================

class SerializationTest : public ::testing::Test {
protected:
    static constexpr double EPSILON = 1e-10;
};

TEST_F(SerializationTest, KBNSumRoundTrip) {
    kbn_sum<double> original;
    original += 1.0;
    original += 2.0;
    original += 3.0;

    auto bytes = to_bytes(original);
    auto restored = from_bytes<kbn_sum<double>>(bytes);

    EXPECT_NEAR(restored.eval(), original.eval(), EPSILON);
}

TEST_F(SerializationTest, CountAccumulatorRoundTrip) {
    count_accumulator original;
    original += 1;
    original += 1;
    original += 1;

    auto bytes = to_bytes(original);
    auto restored = from_bytes<count_accumulator>(bytes);

    EXPECT_EQ(restored.eval(), original.eval());
}

TEST_F(SerializationTest, MinAccumulatorRoundTrip) {
    min_accumulator<double> original;
    original += 5.0;
    original += 3.0;
    original += 7.0;

    auto bytes = to_bytes(original);
    auto restored = from_bytes<min_accumulator<double>>(bytes);

    EXPECT_NEAR(restored.eval(), 3.0, EPSILON);
}

TEST_F(SerializationTest, MaxAccumulatorRoundTrip) {
    max_accumulator<double> original;
    original += 5.0;
    original += 3.0;
    original += 7.0;

    auto bytes = to_bytes(original);
    auto restored = from_bytes<max_accumulator<double>>(bytes);

    EXPECT_NEAR(restored.eval(), 7.0, EPSILON);
}

TEST_F(SerializationTest, ToJson) {
    welford_accumulator<double> acc;
    acc += 1.0;
    acc += 2.0;
    acc += 3.0;

    std::string json = to_json(acc);

    EXPECT_TRUE(json.find("\"value\"") != std::string::npos);
    EXPECT_TRUE(json.find("\"size\":3") != std::string::npos);
    EXPECT_TRUE(json.find("\"mean\"") != std::string::npos);
}

TEST_F(SerializationTest, EmptyAccumulatorSerialization) {
    kbn_sum<double> empty;

    auto bytes = to_bytes(empty);
    auto restored = from_bytes<kbn_sum<double>>(bytes);

    EXPECT_NEAR(restored.eval(), 0.0, EPSILON);
}

TEST_F(SerializationTest, BinaryBufferOperations) {
    binary_buffer buf;

    buf.write<int32_t>(42);
    buf.write<double>(3.14159);
    buf.write_string("hello");

    buf.reset_read();

    EXPECT_EQ(buf.read<int32_t>(), 42);
    EXPECT_NEAR(buf.read<double>(), 3.14159, 1e-10);
    EXPECT_EQ(buf.read_string(), "hello");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

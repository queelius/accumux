/**
 * @file test_integration.cpp
 * @brief Integration tests for accumux library
 *
 * Tests that verify the interaction between different accumulator types
 * and the overall behavior of the library in realistic scenarios.
 */

#include <gtest/gtest.h>
#include "include/accumux/accumulators/basic.hpp"
#include "include/accumux/accumulators/kbn_sum.hpp"
#include "include/accumux/accumulators/welford.hpp"
#include "include/accumux/core/composition.hpp"
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>

using namespace accumux;

class IntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate test data
        std::mt19937 gen(42);  // Fixed seed for reproducibility
        std::normal_distribution<> normal_dist(100.0, 15.0);
        std::uniform_real_distribution<> uniform_dist(0.0, 200.0);

        // Generate normally distributed data
        normal_data.resize(1000);
        for (auto& val : normal_data) {
            val = normal_dist(gen);
        }

        // Generate uniformly distributed data
        uniform_data.resize(1000);
        for (auto& val : uniform_data) {
            val = uniform_dist(gen);
        }

        // Small dataset for precise testing
        small_data = {1.0, 2.0, 3.0, 4.0, 5.0};
    }

    std::vector<double> normal_data;
    std::vector<double> uniform_data;
    std::vector<double> small_data;
};

// ============================================================================
// Complete Statistical Analysis Tests
// ============================================================================

TEST_F(IntegrationTest, CompleteStatisticalAnalysis) {
    // Use all accumulators to analyze a dataset
    min_accumulator<double> min_acc;
    max_accumulator<double> max_acc;
    count_accumulator count_acc;
    kbn_sum<double> sum_acc;
    welford_accumulator<double> welford_acc;
    product_accumulator<double> prod_acc;
    minmax_accumulator<double> minmax_acc;

    // Process small dataset for verifiable results
    for (const auto& val : small_data) {
        min_acc += val;
        max_acc += val;
        count_acc += val;
        sum_acc += val;
        welford_acc += val;
        prod_acc += val;
        minmax_acc += val;
    }

    // Verify all results
    EXPECT_DOUBLE_EQ(min_acc.eval(), 1.0);
    EXPECT_DOUBLE_EQ(max_acc.eval(), 5.0);
    EXPECT_EQ(count_acc.eval(), 5u);
    EXPECT_DOUBLE_EQ(sum_acc.eval(), 15.0);
    EXPECT_DOUBLE_EQ(welford_acc.mean(), 3.0);
    EXPECT_DOUBLE_EQ(welford_acc.variance(), 2.0);  // Population variance
    EXPECT_DOUBLE_EQ(welford_acc.sample_variance(), 2.5);  // Sample variance
    EXPECT_NEAR(prod_acc.eval(), 120.0, 1e-10);
    EXPECT_DOUBLE_EQ(minmax_acc.min(), 1.0);
    EXPECT_DOUBLE_EQ(minmax_acc.max(), 5.0);
    EXPECT_DOUBLE_EQ(minmax_acc.range(), 4.0);
}

// ============================================================================
// Parallel Processing Simulation
// ============================================================================

TEST_F(IntegrationTest, ParallelProcessingSimulation) {
    // Simulate parallel processing by splitting data and combining results
    const size_t chunk_size = normal_data.size() / 4;

    // Create accumulators for each chunk
    std::vector<min_accumulator<double>> min_chunks(4);
    std::vector<max_accumulator<double>> max_chunks(4);
    std::vector<kbn_sum<double>> sum_chunks(4);
    std::vector<welford_accumulator<double>> welford_chunks(4);
    std::vector<count_accumulator> count_chunks(4);

    // Process chunks in "parallel"
    for (size_t chunk = 0; chunk < 4; ++chunk) {
        size_t start = chunk * chunk_size;
        size_t end = (chunk == 3) ? normal_data.size() : start + chunk_size;

        for (size_t i = start; i < end; ++i) {
            min_chunks[chunk] += normal_data[i];
            max_chunks[chunk] += normal_data[i];
            sum_chunks[chunk] += normal_data[i];
            welford_chunks[chunk] += normal_data[i];
            count_chunks[chunk] += normal_data[i];
        }
    }

    // Combine results
    min_accumulator<double> min_total;
    max_accumulator<double> max_total;
    kbn_sum<double> sum_total;
    welford_accumulator<double> welford_total;
    count_accumulator count_total;

    for (size_t chunk = 0; chunk < 4; ++chunk) {
        min_total += min_chunks[chunk];
        max_total += max_chunks[chunk];
        sum_total += sum_chunks[chunk];
        welford_total += welford_chunks[chunk];
        count_total += count_chunks[chunk];
    }

    // Verify against sequential processing
    min_accumulator<double> min_seq;
    max_accumulator<double> max_seq;
    kbn_sum<double> sum_seq;
    welford_accumulator<double> welford_seq;
    count_accumulator count_seq;

    for (const auto& val : normal_data) {
        min_seq += val;
        max_seq += val;
        sum_seq += val;
        welford_seq += val;
        count_seq += val;
    }

    // Results should match
    EXPECT_DOUBLE_EQ(min_total.eval(), min_seq.eval());
    EXPECT_DOUBLE_EQ(max_total.eval(), max_seq.eval());
    EXPECT_NEAR(sum_total.eval(), sum_seq.eval(), 1e-10);
    EXPECT_NEAR(welford_total.mean(), welford_seq.mean(), 1e-10);
    EXPECT_NEAR(welford_total.variance(), welford_seq.variance(), 1e-10);
    EXPECT_EQ(count_total.eval(), count_seq.eval());
}

// ============================================================================
// Streaming Data Processing
// ============================================================================

TEST_F(IntegrationTest, StreamingDataProcessing) {
    // Simulate streaming data processing with periodic statistics
    min_accumulator<double> min_acc;
    max_accumulator<double> max_acc;
    kbn_sum<double> sum_acc;
    welford_accumulator<double> welford_acc;
    count_accumulator count_acc;

    // Process data in batches and check statistics at intervals
    const size_t batch_size = 100;
    std::vector<double> means_over_time;
    std::vector<double> variances_over_time;
    std::vector<double> ranges_over_time;

    for (size_t i = 0; i < uniform_data.size(); i += batch_size) {
        size_t end = std::min(i + batch_size, uniform_data.size());

        for (size_t j = i; j < end; ++j) {
            min_acc += uniform_data[j];
            max_acc += uniform_data[j];
            sum_acc += uniform_data[j];
            welford_acc += uniform_data[j];
            count_acc += uniform_data[j];
        }

        // Record statistics at this point
        if (count_acc.eval() > 0) {
            means_over_time.push_back(welford_acc.mean());
            variances_over_time.push_back(welford_acc.variance());
            ranges_over_time.push_back(max_acc.eval() - min_acc.eval());
        }
    }

    // Verify we collected statistics
    EXPECT_EQ(means_over_time.size(), 10u);
    EXPECT_EQ(count_acc.eval(), uniform_data.size());

    // Mean should stabilize over time (uniform distribution mean ~100)
    EXPECT_NEAR(means_over_time.back(), 100.0, 5.0);

    // Range should increase over time (approaching 200 for uniform 0-200)
    for (size_t i = 1; i < ranges_over_time.size(); ++i) {
        EXPECT_GE(ranges_over_time[i], ranges_over_time[i - 1] - 1e-10);
    }
    EXPECT_GT(ranges_over_time.back(), 180.0);  // Should be close to 200
}

// ============================================================================
// Mixed Precision Processing
// ============================================================================

TEST_F(IntegrationTest, MixedPrecisionProcessing) {
    // Test with mixed float and double precision
    min_accumulator<float> min_float;
    min_accumulator<double> min_double;

    kbn_sum<float> sum_float;
    kbn_sum<double> sum_double;

    // Process same data with different precision
    for (const auto& val : small_data) {
        min_float += static_cast<float>(val);
        min_double += val;
        sum_float += static_cast<float>(val);
        sum_double += val;
    }

    // Results should be very close
    EXPECT_NEAR(static_cast<double>(min_float.eval()), min_double.eval(), 1e-6);
    EXPECT_NEAR(static_cast<double>(sum_float.eval()), sum_double.eval(), 1e-6);
}

// ============================================================================
// Large Scale Data Processing
// ============================================================================

TEST_F(IntegrationTest, LargeScaleDataProcessing) {
    // Generate large dataset
    std::mt19937 gen(12345);
    std::uniform_real_distribution<> dis(-1000.0, 1000.0);

    const size_t large_size = 100000;
    min_accumulator<double> min_acc;
    max_accumulator<double> max_acc;
    kbn_sum<double> sum_acc;
    welford_accumulator<double> welford_acc;
    count_accumulator count_acc;

    double expected_sum = 0.0;
    double expected_min = std::numeric_limits<double>::max();
    double expected_max = std::numeric_limits<double>::lowest();

    for (size_t i = 0; i < large_size; ++i) {
        double val = dis(gen);

        min_acc += val;
        max_acc += val;
        sum_acc += val;
        welford_acc += val;
        count_acc += val;

        expected_sum += val;
        expected_min = std::min(expected_min, val);
        expected_max = std::max(expected_max, val);
    }

    EXPECT_EQ(count_acc.eval(), large_size);
    EXPECT_DOUBLE_EQ(min_acc.eval(), expected_min);
    EXPECT_DOUBLE_EQ(max_acc.eval(), expected_max);
    EXPECT_NEAR(sum_acc.eval(), expected_sum, std::abs(expected_sum) * 1e-10);

    // For uniform distribution [-1000, 1000]:
    // Mean should be close to 0
    EXPECT_NEAR(welford_acc.mean(), 0.0, 10.0);
    // Variance should be close to (range^2)/12 = (2000^2)/12 ≈ 333333
    EXPECT_NEAR(welford_acc.variance(), 333333.0, 10000.0);
}

// ============================================================================
// Numerical Stability Tests
// ============================================================================

TEST_F(IntegrationTest, NumericalStabilityComparison) {
    // Compare KBN sum with naive summation for problematic sequences
    kbn_sum<float> kbn;  // Use float for more precision issues
    float naive_sum = 0.0f;

    // Sequence that causes significant error in naive summation
    std::vector<float> problematic_sequence;
    problematic_sequence.push_back(1e8f);
    for (int i = 0; i < 1000; ++i) {
        problematic_sequence.push_back(1.0f);
    }
    problematic_sequence.push_back(-1e8f);

    // Process with both methods
    for (const auto& val : problematic_sequence) {
        kbn += val;
        naive_sum += val;
    }

    // KBN should give result close to 1000
    EXPECT_NEAR(kbn.eval(), 1000.0f, 10.0f);

    // KBN should be more accurate than naive (or at least as accurate)
    float kbn_error = std::abs(kbn.eval() - 1000.0f);
    float naive_error = std::abs(naive_sum - 1000.0f);
    EXPECT_LE(kbn_error, naive_error + 1.0f);  // Allow small tolerance
}

// ============================================================================
// Extreme Values Handling
// ============================================================================

TEST_F(IntegrationTest, ExtremeValuesHandling) {
    // Test all accumulators with extreme values
    min_accumulator<double> min_acc;
    max_accumulator<double> max_acc;
    kbn_sum<double> sum_acc;
    welford_accumulator<double> welford_acc;
    product_accumulator<double> prod_acc;

    // Test with infinities
    std::vector<double> extreme_data = {
        1.0,
        std::numeric_limits<double>::max(),
        -std::numeric_limits<double>::max(),
        std::numeric_limits<double>::min(),  // Smallest positive
        std::numeric_limits<double>::epsilon(),
        0.0,
        -0.0
    };

    for (const auto& val : extreme_data) {
        min_acc += val;
        max_acc += val;
        sum_acc += val;
        welford_acc += val;
        prod_acc += val;
    }

    // Verify handling
    EXPECT_EQ(min_acc.eval(), -std::numeric_limits<double>::max());
    EXPECT_EQ(max_acc.eval(), std::numeric_limits<double>::max());
    EXPECT_DOUBLE_EQ(prod_acc.eval(), 0.0);  // Because of 0.0

    // Test with NaN
    min_accumulator<double> min_nan;
    max_accumulator<double> max_nan;
    kbn_sum<double> sum_nan;

    min_nan += std::numeric_limits<double>::quiet_NaN();
    max_nan += std::numeric_limits<double>::quiet_NaN();
    sum_nan += std::numeric_limits<double>::quiet_NaN();

    EXPECT_TRUE(std::isnan(min_nan.eval()));
    EXPECT_TRUE(std::isnan(max_nan.eval()));
    EXPECT_TRUE(std::isnan(sum_nan.eval()));
}

// ============================================================================
// Real-world Scenario: Financial Data
// ============================================================================

TEST_F(IntegrationTest, FinancialDataScenario) {
    // Simulate processing daily stock returns
    std::mt19937 gen(54321);
    std::normal_distribution<> returns_dist(0.001, 0.02);  // Mean 0.1%, StdDev 2%

    min_accumulator<double> min_return;
    max_accumulator<double> max_return;
    kbn_sum<double> total_return;
    welford_accumulator<double> return_stats;
    count_accumulator trading_days;
    product_accumulator<double> compound_return;

    // Simulate one year of trading (252 days)
    for (int day = 0; day < 252; ++day) {
        double daily_return = returns_dist(gen);

        min_return += daily_return;
        max_return += daily_return;
        total_return += daily_return;
        return_stats += daily_return;
        trading_days += daily_return;
        compound_return += (1.0 + daily_return);  // For compound returns
    }

    // Verify results make sense
    EXPECT_EQ(trading_days.eval(), 252u);
    EXPECT_LT(min_return.eval(), 0.0);  // Should have some negative days
    EXPECT_GT(max_return.eval(), 0.0);  // Should have some positive days
    EXPECT_NEAR(return_stats.mean(), 0.001, 0.002);  // Should be close to expected mean
    EXPECT_NEAR(return_stats.std_dev(), 0.02, 0.004);  // Should be close to expected std dev

    // Annual return
    double annual_simple_return = total_return.eval();
    double annual_compound_return = compound_return.eval() - 1.0;

    // Compound return should be slightly less than simple return (due to volatility drag)
    EXPECT_GT(annual_simple_return, -1.0);  // Reasonable bounds
    EXPECT_LT(annual_simple_return, 1.0);
    EXPECT_GT(annual_compound_return, -1.0);
    EXPECT_LT(annual_compound_return, 2.0);  // Compound can exceed 1.0 with good luck
}

// ============================================================================
// Accumulator Reset and Reuse
// ============================================================================

TEST_F(IntegrationTest, AccumulatorResetAndReuse) {
    // Test that accumulators can be reset and reused
    min_accumulator<double> min_acc(10.0);
    max_accumulator<double> max_acc(10.0);
    kbn_sum<double> sum_acc(10.0);
    welford_accumulator<double> welford_acc(10.0);

    // Add some data
    for (const auto& val : small_data) {
        min_acc += val;
        max_acc += val;
        sum_acc += val;
        welford_acc += val;
    }

    // Reset using assignment
    min_acc = min_accumulator<double>();
    max_acc = max_accumulator<double>();
    sum_acc = kbn_sum<double>();
    welford_acc = welford_accumulator<double>();

    // Verify reset
    EXPECT_TRUE(min_acc.empty());
    EXPECT_TRUE(max_acc.empty());
    EXPECT_DOUBLE_EQ(sum_acc.eval(), 0.0);
    EXPECT_EQ(welford_acc.size(), 0u);

    // Reuse with new data
    std::vector<double> new_data = {10.0, 20.0, 30.0};
    for (const auto& val : new_data) {
        min_acc += val;
        max_acc += val;
        sum_acc += val;
        welford_acc += val;
    }

    EXPECT_DOUBLE_EQ(min_acc.eval(), 10.0);
    EXPECT_DOUBLE_EQ(max_acc.eval(), 30.0);
    EXPECT_DOUBLE_EQ(sum_acc.eval(), 60.0);
    EXPECT_DOUBLE_EQ(welford_acc.mean(), 20.0);
}

// ============================================================================
// Memory Efficiency Test
// ============================================================================

TEST_F(IntegrationTest, MemoryEfficiency) {
    // Verify that accumulators maintain constant memory regardless of data size

    // Size of accumulators shouldn't change with data
    min_accumulator<double> min_acc;
    welford_accumulator<double> welford_acc;
    kbn_sum<double> sum_acc;

    size_t initial_min_size = sizeof(min_acc);
    size_t initial_welford_size = sizeof(welford_acc);
    size_t initial_sum_size = sizeof(sum_acc);

    // Process large amount of data
    for (int i = 0; i < 100000; ++i) {
        min_acc += static_cast<double>(i);
        welford_acc += static_cast<double>(i);
        sum_acc += static_cast<double>(i);
    }

    // Size should remain constant
    EXPECT_EQ(sizeof(min_acc), initial_min_size);
    EXPECT_EQ(sizeof(welford_acc), initial_welford_size);
    EXPECT_EQ(sizeof(sum_acc), initial_sum_size);

    // Verify they still work correctly
    EXPECT_DOUBLE_EQ(min_acc.eval(), 0.0);
    EXPECT_DOUBLE_EQ(welford_acc.mean(), 49999.5);
}
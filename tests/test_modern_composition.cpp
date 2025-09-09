/**
 * @file test_modern_composition.cpp
 * @brief Comprehensive tests for the modern composition system
 */

#include <gtest/gtest.h>
#include "include/accumux/core/composition.hpp"
#include "include/accumux/accumulators/kbn_sum.hpp"
#include "include/accumux/accumulators/welford.hpp"
#include "include/accumux/accumulators/basic.hpp"
#include <vector>
#include <random>
#include <algorithm>

using namespace accumux;

class ModernCompositionTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_data = {1.0, 2.0, 3.0, 4.0, 5.0};
        expected_sum = 15.0;
        expected_mean = 3.0;
        expected_variance = 2.5; // Population variance
        expected_min = 1.0;
        expected_max = 5.0;
        expected_count = 5;
    }
    
    std::vector<double> test_data;
    double expected_sum;
    double expected_mean;
    double expected_variance;
    double expected_min;
    double expected_max;
    std::size_t expected_count;
};

// ============================================================================
// Concept Compliance Tests
// ============================================================================

TEST_F(ModernCompositionTest, ConceptCompliance_BasicAccumulators) {
    // Test that all new accumulators satisfy the Accumulator concept
    static_assert(Accumulator<kbn_sum<double>>);
    static_assert(Accumulator<welford_accumulator<double>>);
    static_assert(Accumulator<min_accumulator<double>>);
    static_assert(Accumulator<max_accumulator<double>>);
    static_assert(Accumulator<count_accumulator>);
    
    // Test statistical concepts
    static_assert(StatisticalAccumulator<welford_accumulator<double>>);
    static_assert(VarianceAccumulator<welford_accumulator<double>>);
    
    SUCCEED();
}

TEST_F(ModernCompositionTest, ConceptCompliance_Compositions) {
    // Test that compositions satisfy the Accumulator concept
    auto composed = kbn_sum<double>() + welford_accumulator<double>();
    static_assert(Accumulator<decltype(composed)>);
    
    SUCCEED();
}

// ============================================================================
// Individual Accumulator Tests
// ============================================================================

TEST_F(ModernCompositionTest, KbnSum_BasicFunctionality) {
    kbn_sum<double> sum_acc;
    
    for (const auto& value : test_data) {
        sum_acc += value;
    }
    
    EXPECT_DOUBLE_EQ(sum_acc.eval(), expected_sum);
    EXPECT_DOUBLE_EQ(static_cast<double>(sum_acc), expected_sum);
}

TEST_F(ModernCompositionTest, WelfordAccumulator_BasicFunctionality) {
    welford_accumulator<double> stats_acc;
    
    for (const auto& value : test_data) {
        stats_acc += value;
    }
    
    EXPECT_DOUBLE_EQ(stats_acc.mean(), expected_mean);
    EXPECT_NEAR(stats_acc.variance(), expected_variance, 1e-10);
    EXPECT_EQ(stats_acc.size(), expected_count);
    EXPECT_DOUBLE_EQ(stats_acc.eval(), expected_mean); // eval() returns mean
}

TEST_F(ModernCompositionTest, BasicAccumulators_Functionality) {
    min_accumulator<double> min_acc;
    max_accumulator<double> max_acc;
    count_accumulator count_acc;
    
    for (const auto& value : test_data) {
        min_acc += value;
        max_acc += value;
        count_acc += value;
    }
    
    EXPECT_DOUBLE_EQ(min_acc.eval(), expected_min);
    EXPECT_DOUBLE_EQ(max_acc.eval(), expected_max);
    EXPECT_EQ(count_acc.eval(), expected_count);
}

// ============================================================================
// Parallel Composition Tests  
// ============================================================================

TEST_F(ModernCompositionTest, ParallelComposition_SumAndStats) {
    auto composed = kbn_sum<double>() + welford_accumulator<double>();
    
    for (const auto& value : test_data) {
        composed += value;
    }
    
    // Test access methods
    const auto& sum_part = composed.get_first();
    const auto& stats_part = composed.get_second();
    
    EXPECT_DOUBLE_EQ(sum_part.eval(), expected_sum);
    EXPECT_DOUBLE_EQ(stats_part.mean(), expected_mean);
    EXPECT_NEAR(stats_part.variance(), expected_variance, 1e-10);
}

TEST_F(ModernCompositionTest, ParallelComposition_MinMaxCount) {
    // Use a specialized minmax accumulator instead of composing min + max
    auto composed = minmax_accumulator<double>() + count_accumulator();
    
    for (const auto& value : test_data) {
        composed += value;
    }
    
    // Access the results
    const auto& minmax_part = composed.get_first();
    const auto& count_part = composed.get_second();
    
    EXPECT_DOUBLE_EQ(minmax_part.min(), expected_min);
    EXPECT_DOUBLE_EQ(minmax_part.max(), expected_max);
    EXPECT_EQ(count_part.eval(), expected_count);
}

TEST_F(ModernCompositionTest, ParallelComposition_EvalTuple) {
    auto composed = kbn_sum<double>() + welford_accumulator<double>();
    
    for (const auto& value : test_data) {
        composed += value;
    }
    
    auto results = composed.eval();
    EXPECT_DOUBLE_EQ(std::get<0>(results), expected_sum);
    EXPECT_DOUBLE_EQ(std::get<1>(results), expected_mean);
}

TEST_F(ModernCompositionTest, ParallelComposition_AccumulatorCombination) {
    auto comp1 = kbn_sum<double>() + welford_accumulator<double>();
    auto comp2 = kbn_sum<double>() + welford_accumulator<double>();
    
    // Split data between two compositions
    comp1 += test_data[0];
    comp1 += test_data[1];
    comp1 += test_data[2];
    
    comp2 += test_data[3];
    comp2 += test_data[4];
    
    // Combine them
    comp1 += comp2;
    
    // Check combined results
    EXPECT_DOUBLE_EQ(comp1.get_first().eval(), expected_sum);
    EXPECT_DOUBLE_EQ(comp1.get_second().mean(), expected_mean);
    EXPECT_NEAR(comp1.get_second().variance(), expected_variance, 1e-10);
}

// ============================================================================
// Type Safety and Template Tests
// ============================================================================

TEST_F(ModernCompositionTest, TypeSafety_ValueTypeDeduction) {
    auto composed = kbn_sum<double>() + welford_accumulator<double>();
    
    using value_t = typename decltype(composed)::value_type;
    static_assert(std::is_same_v<value_t, double>);
    
    SUCCEED();
}

TEST_F(ModernCompositionTest, TypeSafety_MixedTypes) {
    // Test composition with different but compatible types
    auto composed = kbn_sum<double>() + min_accumulator<double>();
    
    for (const auto& value : test_data) {
        composed += value;
    }
    
    EXPECT_DOUBLE_EQ(composed.get_first().eval(), expected_sum);
    EXPECT_DOUBLE_EQ(composed.get_second().eval(), expected_min);
}

// ============================================================================
// Performance and Efficiency Tests
// ============================================================================

TEST_F(ModernCompositionTest, Performance_LargeDataset) {
    auto composed = kbn_sum<double>() + welford_accumulator<double>();
    
    // Generate large dataset
    std::vector<double> large_data(100000);
    std::iota(large_data.begin(), large_data.end(), 1.0);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (const auto& value : large_data) {
        composed += value;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Should complete reasonably quickly (this is a smoke test)
    EXPECT_LT(duration.count(), 100); // Less than 100ms for 100k elements
    
    // Verify results
    double expected_large_sum = large_data.size() * (large_data.size() + 1) / 2.0;
    EXPECT_DOUBLE_EQ(composed.get_first().eval(), expected_large_sum);
    EXPECT_EQ(composed.get_second().size(), large_data.size());
}

// ============================================================================
// Complex Composition Scenarios
// ============================================================================

TEST_F(ModernCompositionTest, ComplexComposition_FinancialAnalysis) {
    // Simulate a financial analysis: sum, mean, variance, min, max
    auto financial_stats = kbn_sum<double>() + welford_accumulator<double>();
    
    // Sample financial data (returns)
    std::vector<double> returns = {0.05, -0.02, 0.03, 0.01, -0.01, 0.04, -0.03, 0.02};
    
    for (const auto& ret : returns) {
        financial_stats += ret;
    }
    
    auto sum_acc = financial_stats.get_first();
    auto welford_part = financial_stats.get_second();
    
    // Verify financial metrics
    EXPECT_NEAR(sum_acc.eval(), 0.09, 1e-10);                    // Total return
    EXPECT_NEAR(welford_part.mean(), 0.01125, 1e-10);           // Average return
    EXPECT_GT(welford_part.sample_variance(), 0);                // Volatility > 0
    EXPECT_EQ(welford_part.size(), returns.size());             // Sample count
}

TEST_F(ModernCompositionTest, ComplexComposition_QualityControl) {
    // Quality control scenario: track statistics and outliers
    auto qc_stats = welford_accumulator<double>() + minmax_accumulator<double>();
    
    // Simulated measurement data with an outlier
    std::vector<double> measurements = {10.1, 10.0, 9.9, 10.2, 10.1, 15.0, 9.8, 10.0};
    
    for (const auto& measurement : measurements) {
        qc_stats += measurement;
    }
    
    auto stats = qc_stats.get_first();
    auto minmax = qc_stats.get_second();
    
    EXPECT_EQ(stats.size(), measurements.size());
    EXPECT_GT(stats.mean(), 10.0);                    // Mean affected by outlier
    EXPECT_GT(stats.sample_variance(), 1.0);          // High variance due to outlier
    EXPECT_EQ(minmax.min(), 9.8);                     // Minimum
    EXPECT_EQ(minmax.max(), 15.0);                    // Maximum (outlier)
    EXPECT_GT(minmax.range(), 5.0);                   // Large range indicates outlier
}

// ============================================================================
// Edge Cases and Robustness Tests
// ============================================================================

TEST_F(ModernCompositionTest, EdgeCase_EmptyData) {
    auto composed = kbn_sum<double>() + welford_accumulator<double>();
    
    // Don't add any data - test default states
    EXPECT_DOUBLE_EQ(composed.get_first().eval(), 0.0);
    EXPECT_DOUBLE_EQ(composed.get_second().mean(), 0.0);
    EXPECT_EQ(composed.get_second().size(), 0);
}

TEST_F(ModernCompositionTest, EdgeCase_SingleValue) {
    auto composed = kbn_sum<double>() + welford_accumulator<double>();
    
    composed += 42.0;
    
    EXPECT_DOUBLE_EQ(composed.get_first().eval(), 42.0);
    EXPECT_DOUBLE_EQ(composed.get_second().mean(), 42.0);
    EXPECT_DOUBLE_EQ(composed.get_second().variance(), 0.0); // No variance with single value
    EXPECT_EQ(composed.get_second().size(), 1);
}

TEST_F(ModernCompositionTest, EdgeCase_ExtremeValues) {
    auto composed = kbn_sum<double>() + welford_accumulator<double>();
    
    std::vector<double> extreme_values = {1e-10, 1e10, -1e10, 1e-15};
    
    for (const auto& value : extreme_values) {
        composed += value;
    }
    
    // Should handle extreme values gracefully
    auto sum_result = composed.get_first().eval();
    auto stats = composed.get_second();
    
    EXPECT_TRUE(std::isfinite(sum_result));
    EXPECT_TRUE(std::isfinite(stats.mean()));
    EXPECT_TRUE(std::isfinite(stats.variance()));
}
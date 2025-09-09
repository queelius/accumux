/**
 * @file test_composition_simple.cpp
 * @brief Simple unit tests for accumulator composition operations
 */

#include <gtest/gtest.h>
#include "include/accumux/core/composition.hpp"
#include "kbn_sum.hpp"
#include "welford_accumulator.hpp"
#include <vector>
#include <chrono>
#include <numeric>

using namespace accumux;
using algebraic_accumulator::kbn_sum;
using algebraic_accumulators::welford_accumulator;

// Type aliases for convenience
using sum_acc = kbn_sum<double>;
using stats_acc = welford_accumulator<kbn_sum<double>>;

class SimpleCompositionTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_data = {1.0, 2.0, 3.0, 4.0, 5.0};
        expected_sum = 15.0;
        expected_mean = 3.0;
    }
    
    std::vector<double> test_data;
    double expected_sum;
    double expected_mean;
};

// ============================================================================
// Basic Parallel Composition Tests
// ============================================================================

TEST_F(SimpleCompositionTest, ParallelComposition_BasicConstruction) {
    // Test that we can construct parallel composition
    sum_acc sum_acc;
    stats_acc stat_acc;
    
    auto composed = parallel_composition<sum_acc, stats_acc>(
        std::move(sum_acc), std::move(stat_acc)
    );
    
    SUCCEED(); // If we get here, construction worked
}

TEST_F(SimpleCompositionTest, ParallelComposition_DataProcessing) {
    sum_acc sum_acc;
    stats_acc stat_acc;
    
    auto composed = parallel_composition<sum_acc, stats_acc>(
        std::move(sum_acc), std::move(stat_acc)
    );
    
    // Add test data one by one
    for (const auto& value : test_data) {
        composed += value;
    }
    
    // Check results using index access
    const auto& sum_result = composed.get_first();
    const auto& stat_result = composed.get_second();
    
    EXPECT_DOUBLE_EQ(sum_result.eval(), expected_sum);
    EXPECT_DOUBLE_EQ(stat_result.mean(), expected_mean);
}

TEST_F(SimpleCompositionTest, ParallelComposition_EvalTuple) {
    sum_acc sum_acc;
    stats_acc stat_acc;
    
    auto composed = parallel_composition<sum_acc, stats_acc>(
        std::move(sum_acc), std::move(stat_acc)
    );
    
    for (const auto& value : test_data) {
        composed += value;
    }
    
    // Test eval() returns tuple
    auto results = composed.eval();
    
    EXPECT_DOUBLE_EQ(std::get<0>(results), expected_sum);
    EXPECT_DOUBLE_EQ(std::get<1>(results), expected_mean); // welford eval returns mean
}

// ============================================================================
// Operator Overload Tests  
// ============================================================================

TEST_F(SimpleCompositionTest, OperatorPlus_BasicUsage) {
    sum_acc sum_acc;
    stats_acc stat_acc;
    
    // Test the + operator
    auto composed = sum_acc + stat_acc;
    
    for (const auto& value : test_data) {
        composed += value;
    }
    
    // Verify it works
    const auto& sum_result = composed.get_first();
    const auto& stat_result = composed.get_second();
    
    EXPECT_DOUBLE_EQ(sum_result.eval(), expected_sum);
    EXPECT_DOUBLE_EQ(stat_result.mean(), expected_mean);
}

// ============================================================================
// Type Safety Tests
// ============================================================================

TEST_F(SimpleCompositionTest, TypeSafety_ValueTypeDeduction) {
    sum_acc sum_acc;
    stats_acc stat_acc;
    
    auto composed = sum_acc + stat_acc;
    
    // Test that value_type is correctly deduced
    using composition_type = decltype(composed);
    using value_t = typename composition_type::value_type;
    static_assert(std::is_same_v<value_t, double>);
    
    SUCCEED();
}

// ============================================================================
// Performance Test
// ============================================================================

TEST_F(SimpleCompositionTest, Performance_BasicEfficiency) {
    sum_acc sum_acc;
    stats_acc stat_acc;
    
    auto composed = sum_acc + stat_acc;
    
    // Large dataset
    std::vector<double> large_data(1000);
    std::iota(large_data.begin(), large_data.end(), 1.0);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (const auto& value : large_data) {
        composed += value;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Should complete reasonably quickly
    EXPECT_LT(duration.count(), 5000); // Less than 5ms for 1k elements
    
    // Verify correctness
    double expected_large_sum = large_data.size() * (large_data.size() + 1) / 2.0;
    EXPECT_DOUBLE_EQ(composed.get_first().eval(), expected_large_sum);
}
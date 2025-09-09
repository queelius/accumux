/**
 * @file test_composition.cpp
 * @brief Unit tests for accumulator composition operations
 */

#include <gtest/gtest.h>
#include "include/accumux/core/composition.hpp"
#include "kbn_sum.hpp"
#include "welford_accumulator.hpp"
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>

using namespace accumux;
using namespace algebraic_accumulator;
using namespace algebraic_accumulators;

class CompositionTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_data = {1.0, 2.0, 3.0, 4.0, 5.0};
        expected_sum = 15.0;
        expected_mean = 3.0;
        expected_variance = 2.5; // Population variance: sum((x-mean)²)/n
    }
    
    std::vector<double> test_data;
    double expected_sum;
    double expected_mean;
    double expected_variance;
};

// ============================================================================
// Parallel Composition Tests (a + b)
// ============================================================================

TEST_F(CompositionTest, ParallelComposition_BasicConstruction) {
    auto composed = kbn_sum<double>() + welford_accumulator<double>();
    
    // Test that we can construct and it compiles
    SUCCEED();
}

TEST_F(CompositionTest, ParallelComposition_DataProcessing) {
    auto composed = kbn_sum<double>() + welford_accumulator<double>();
    
    // Add all test data
    for (const auto& value : test_data) {
        composed += value;
    }
    
    // Check that both accumulators computed correctly
    auto sum_result = composed.get<kbn_sum<double>>();
    auto welford_result = composed.get<welford_accumulator<double>>();
    
    EXPECT_DOUBLE_EQ(sum_result.eval(), expected_sum);
    EXPECT_DOUBLE_EQ(welford_result.mean(), expected_mean);
    EXPECT_NEAR(welford_result.variance(), expected_variance, 1e-10);
}

TEST_F(CompositionTest, ParallelComposition_IndexAccess) {
    auto composed = kbn_sum<double>() + welford_accumulator<double>();
    
    for (const auto& value : test_data) {
        composed += value;
    }
    
    // Test index-based access
    auto sum_by_index = composed.get<0>();
    auto welford_by_index = composed.get<1>();
    
    EXPECT_DOUBLE_EQ(sum_by_index.eval(), expected_sum);
    EXPECT_DOUBLE_EQ(welford_by_index.mean(), expected_mean);
}

TEST_F(CompositionTest, ParallelComposition_EvalTuple) {
    auto composed = kbn_sum<double>() + welford_accumulator<double>();
    
    for (const auto& value : test_data) {
        composed += value;
    }
    
    // Test eval() returns tuple
    auto results = composed.eval();
    
    EXPECT_DOUBLE_EQ(std::get<0>(results), expected_sum);
    EXPECT_DOUBLE_EQ(std::get<1>(results), expected_mean); // welford eval returns mean
}

TEST_F(CompositionTest, ParallelComposition_AccumulatorCombination) {
    auto composed1 = kbn_sum<double>() + welford_accumulator<double>();
    auto composed2 = kbn_sum<double>() + welford_accumulator<double>();
    
    // Add first half to composed1
    composed1 += test_data[0];
    composed1 += test_data[1];
    composed1 += test_data[2];
    
    // Add second half to composed2
    composed2 += test_data[3];
    composed2 += test_data[4];
    
    // Combine them
    composed1 += composed2;
    
    // Check combined results
    auto sum_result = composed1.get<kbn_sum<double>>();
    auto welford_result = composed1.get<welford_accumulator<double>>();
    
    EXPECT_DOUBLE_EQ(sum_result.eval(), expected_sum);
    EXPECT_DOUBLE_EQ(welford_result.mean(), expected_mean);
}

TEST_F(CompositionTest, ParallelComposition_ThreeAccumulators) {
    // Test composition with more accumulators
    auto sum1 = kbn_sum<double>();
    auto sum2 = kbn_sum<double>();
    auto stats = welford_accumulator<double>();
    
    // Compose: (sum1 + sum2) + stats
    auto composed = (sum1 + sum2) + stats;
    
    for (const auto& value : test_data) {
        composed += value;
    }
    
    // The outer composition should work
    SUCCEED();
}

// ============================================================================
// Sequential Composition Tests (a * b) 
// ============================================================================

TEST_F(CompositionTest, SequentialComposition_BasicConstruction) {
    auto pipeline = kbn_sum<double>() * welford_accumulator<double>();
    
    // Test construction compiles
    SUCCEED();
}

TEST_F(CompositionTest, SequentialComposition_DataFlow) {
    auto pipeline = kbn_sum<double>() * welford_accumulator<double>();
    
    // Note: This is conceptual - the actual behavior depends on how
    // we define sequential semantics. For now, test basic functionality.
    pipeline += 1.0;
    pipeline += 2.0;
    
    // Test that we can call eval
    auto result = pipeline.eval();
    auto intermediate = pipeline.intermediate();
    
    // Basic smoke test - detailed semantics need refinement
    SUCCEED();
}

// ============================================================================
// Conditional Composition Tests
// ============================================================================

TEST_F(CompositionTest, ConditionalComposition_PredicateSwitch) {
    // Use different accumulators based on value size
    auto small_handler = kbn_sum<double>();
    auto large_handler = welford_accumulator<double>();
    
    // Predicate: use first accumulator for values < 3
    auto predicate = [](double x) { return x < 3.0; };
    auto conditional_comp = conditional(std::move(small_handler), std::move(large_handler), predicate);
    
    // Add values that will trigger both accumulators
    conditional_comp += 1.0; // Goes to first (< 3)
    conditional_comp += 2.0; // Goes to first (< 3)  
    conditional_comp += 4.0; // Goes to second (>= 3)
    conditional_comp += 5.0; // Goes to second (>= 3)
    
    // Test that it completes without error
    auto result = conditional_comp.eval();
    SUCCEED();
}

// ============================================================================
// Expression Template Style Tests
// ============================================================================

TEST_F(CompositionTest, ExpressionSyntax_Natural) {
    // Test that we can write natural mathematical expressions
    auto computation = kbn_sum<double>() + welford_accumulator<double>();
    
    // Process data with natural syntax
    std::for_each(test_data.begin(), test_data.end(), 
                  [&](double x) { computation += x; });
    
    // Verify results
    EXPECT_DOUBLE_EQ(computation.get<kbn_sum<double>>().eval(), expected_sum);
    EXPECT_DOUBLE_EQ(computation.get<welford_accumulator<double>>().mean(), expected_mean);
}

TEST_F(CompositionTest, ExpressionSyntax_Chaining) {
    // Test chaining compositions
    auto base_sum = kbn_sum<double>();
    auto stats = welford_accumulator<double>();
    
    auto composed = base_sum + stats;
    
    // Chain data processing
    auto result = std::accumulate(test_data.begin(), test_data.end(), 
                                std::move(composed),
                                [](auto&& acc, double x) -> auto&& {
                                    acc += x;
                                    return std::forward<decltype(acc)>(acc);
                                });
    
    EXPECT_DOUBLE_EQ(result.get<kbn_sum<double>>().eval(), expected_sum);
}

// ============================================================================
// Type Safety and Concept Tests  
// ============================================================================

TEST_F(CompositionTest, TypeSafety_ConceptCompliance) {
    // Test that composed accumulators satisfy Accumulator concept
    auto composed = kbn_sum<double>() + welford_accumulator<double>();
    
    // These should compile if the concept is satisfied
    static_assert(Accumulator<decltype(composed)>);
    
    // Test value_type is properly defined
    using value_t = typename decltype(composed)::value_type;
    static_assert(std::is_same_v<value_t, double>);
    
    SUCCEED();
}

TEST_F(CompositionTest, TypeSafety_InvalidTypeAccess) {
    auto composed = kbn_sum<double>() + welford_accumulator<double>();
    
    // This should not compile (commented out):
    // auto invalid = composed.get<int>(); // Should fail static_assert
    
    SUCCEED();
}

// ============================================================================
// Performance and Efficiency Tests
// ============================================================================

TEST_F(CompositionTest, Performance_SinglePassEfficiency) {
    auto composed = kbn_sum<double>() + welford_accumulator<double>();
    
    // Large dataset to test efficiency
    std::vector<double> large_data(10000);
    std::iota(large_data.begin(), large_data.end(), 1.0);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (const auto& value : large_data) {
        composed += value;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Should complete reasonably quickly (this is more of a smoke test)
    EXPECT_LT(duration.count(), 10000); // Less than 10ms for 10k elements
    
    // Verify results are still correct
    double expected_large_sum = large_data.size() * (large_data.size() + 1) / 2.0;
    EXPECT_DOUBLE_EQ(composed.get<kbn_sum<double>>().eval(), expected_large_sum);
}
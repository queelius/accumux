#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <limits>
#include "include/accumux/accumulators/kbn_sum.hpp"
#include "include/accumux/accumulators/welford.hpp"

using namespace accumux;

class UtilityCoverageTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test the abs function for kbn_sum
TEST_F(UtilityCoverageTest, KBNSumAbsFunction) {
    kbn_sum<double> positive(5.5);
    // We can't directly modify correction anymore
    
    auto abs_result = abs(positive);
    EXPECT_EQ(static_cast<double>(abs_result), 5.5);
    
    // Test with negative values
    kbn_sum<double> negative(-3.2);
    
    auto abs_negative = abs(negative);
    EXPECT_EQ(static_cast<double>(abs_negative), 3.2);
    
    // Test with zero
    kbn_sum<double> zero(0.0);
    
    auto abs_zero = abs(zero);
    EXPECT_EQ(static_cast<double>(abs_zero), 0.0);
}

// Test float specialization branch coverage for KBN sum
TEST_F(UtilityCoverageTest, KBNSumFloatBranchCoverage) {
    kbn_sum<float> sum(1.0f);
    
    // Test accumulation with float precision
    sum = kbn_sum<float>(10.0f);
    sum += 1.0f; // abs(1.0f) < abs(10.0f), so should hit first branch
    
    EXPECT_GT(static_cast<float>(sum), 10.5f);
    
    // Test the else branch for float: abs(x) >= abs(s)
    kbn_sum<float> sum2(1.0f);
    sum2 += 10.0f; // abs(10.0f) >= abs(1.0f), should hit else branch
    
    EXPECT_EQ(static_cast<float>(sum2), 11.0f);
}

// Test uncovered Welford accumulator methods
TEST_F(UtilityCoverageTest, WelfordUtilityFunctions) {
    welford_accumulator<double> acc;
    
    // Add some data points
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    for (double val : data) {
        acc += val;
    }
    
    // Test variance method directly
    double var = acc.variance();
    EXPECT_DOUBLE_EQ(var, 2.0);
    
    // Test sample_variance method directly  
    double sample_var = acc.sample_variance();
    EXPECT_DOUBLE_EQ(sample_var, 2.5);
    
    // Test sum method directly
    double total = acc.sum();
    EXPECT_DOUBLE_EQ(total, 15.0);
    
    // These are methods on the accumulator, not standalone functions
    EXPECT_DOUBLE_EQ(acc.mean(), 3.0);
    EXPECT_DOUBLE_EQ(acc.variance(), 2.0);
    EXPECT_DOUBLE_EQ(acc.sample_variance(), 2.5);
    EXPECT_EQ(acc.size(), 5u);
    EXPECT_DOUBLE_EQ(acc.sum(), 15.0);
    
    // Test conversion operator (returns mean)
    double converted_value = static_cast<double>(acc);
    EXPECT_DOUBLE_EQ(converted_value, 3.0);
}

// Test value constructor for Welford accumulator
TEST_F(UtilityCoverageTest, WelfordValueConstructor) {
    welford_accumulator<double> acc(10.0);
    
    EXPECT_EQ(acc.size(), 1u);
    EXPECT_DOUBLE_EQ(acc.mean(), 10.0);
    EXPECT_DOUBLE_EQ(acc.sum(), 10.0);
    
    // Add another value to test it works properly
    acc += 20.0;
    EXPECT_EQ(acc.size(), 2u);
    EXPECT_DOUBLE_EQ(acc.mean(), 15.0);
    EXPECT_DOUBLE_EQ(acc.sum(), 30.0);
}

// Test KBN sum eval method
TEST_F(UtilityCoverageTest, KBNSumEvalMethod) {
    kbn_sum<double> sum(5.0);
    
    EXPECT_DOUBLE_EQ(sum.eval(), 5.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(sum), sum.eval());
}

// Test KBN sum comparison with scalar
TEST_F(UtilityCoverageTest, KBNSumScalarComparison) {
    kbn_sum<double> sum(5.0);
    
    EXPECT_TRUE(sum < 6.0);
    EXPECT_FALSE(sum < 5.0);
    EXPECT_TRUE(sum < 5.5);
}

// Test KBN sum assignment operator
TEST_F(UtilityCoverageTest, KBNSumValueAssignment) {
    kbn_sum<double> sum(10.0);
    
    sum = 3.0;
    EXPECT_DOUBLE_EQ(sum.sum_component(), 3.0);
    EXPECT_DOUBLE_EQ(sum.correction_component(), 0.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(sum), 3.0);
}

// Test edge cases with very small and large numbers
TEST_F(UtilityCoverageTest, ExtremePrecisionCases) {
    kbn_sum<double> sum;
    
    // Add very large number followed by very small ones
    sum += 1e16;
    sum += 1.0;
    sum += 1.0;
    sum += -1e16;
    
    // KBN sum should preserve the 2.0 better than naive summation
    double result = static_cast<double>(sum);
    EXPECT_NEAR(result, 2.0, 1e-10);
    EXPECT_GT(result, 1.9); // Should be better than total loss of precision
}

// Test Welford accumulator with edge cases
TEST_F(UtilityCoverageTest, WelfordEdgeCases) {
    welford_accumulator<double> acc;
    
    // Single value case
    acc += 42.0;
    EXPECT_DOUBLE_EQ(acc.mean(), 42.0);
    EXPECT_DOUBLE_EQ(acc.variance(), 0.0);
    
    // Two identical values
    acc += 42.0;
    EXPECT_DOUBLE_EQ(acc.mean(), 42.0);
    EXPECT_DOUBLE_EQ(acc.variance(), 0.0);
    EXPECT_DOUBLE_EQ(acc.sample_variance(), 0.0);
}

// Test using factory function
TEST_F(UtilityCoverageTest, FactoryFunctions) {
    auto kbn_acc = make_kbn_sum<double>(10.0);
    auto welford_acc = make_welford_accumulator<double>();
    
    welford_acc += 1.0;
    welford_acc += 2.0;
    welford_acc += 3.0;
    
    EXPECT_EQ(welford_acc.size(), 3u);
    EXPECT_DOUBLE_EQ(welford_acc.mean(), 2.0);
    EXPECT_DOUBLE_EQ(welford_acc.sum(), 6.0);
    EXPECT_EQ(static_cast<double>(kbn_acc), 10.0);
}

// Test zero and negative operations more thoroughly
TEST_F(UtilityCoverageTest, ZeroAndNegativeOperations) {
    kbn_sum<double> sum;
    
    // Test operations with zero
    sum += 0.0;
    EXPECT_DOUBLE_EQ(static_cast<double>(sum), 0.0);
    
    sum += -0.0;
    EXPECT_DOUBLE_EQ(static_cast<double>(sum), 0.0);
    
    // Test negative cancellation
    sum = 5.0;
    sum += -5.0;
    EXPECT_DOUBLE_EQ(static_cast<double>(sum), 0.0);
}
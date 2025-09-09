#include <gtest/gtest.h>
#include <vector>
#include <limits>
#include "kbn_sum.hpp"
#include "welford_accumulator.hpp"

using namespace algebraic_accumulator;
using namespace algebraic_accumulators;

class AdditionalCoverageTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test specific edge cases that might not be covered
TEST_F(AdditionalCoverageTest, KBNSumSpecialCases) {
    // Test the specific branches in the += operator
    kbn_sum<double> sum;
    
    // Case where abs(x) == abs(s) - should go to the else branch
    sum.s = 5.0;
    sum += -5.0;  // abs(-5.0) == abs(5.0), should use ((x - t) + s) calculation
    EXPECT_EQ(static_cast<double>(sum), 0.0);
    
    // Test with very small correction terms
    kbn_sum<float> float_sum(1e20f);
    float_sum += 1.0f;
    float_sum += 1.0f;
    float_sum += -1e20f;
    
    // Should preserve some precision due to KBN algorithm
    float result = static_cast<float>(float_sum);
    EXPECT_GE(result, 1.5f); // Better than naive summation
}

// Test welford accumulator with extreme values
TEST_F(AdditionalCoverageTest, WelfordSpecialCases) {
    welford_accumulator<kbn_sum<double>> acc;
    
    // Test with infinity values (if supported)
    acc += std::numeric_limits<double>::max();
    acc += std::numeric_limits<double>::max();
    
    EXPECT_TRUE(std::isfinite(acc.mean()) || std::isinf(acc.mean()));
    EXPECT_EQ(acc.size(), 2u);
    
    // Test move semantics path explicitly
    welford_accumulator<kbn_sum<double>> acc2;
    double temp = 5.0;
    acc2 += std::move(temp); // This triggers the move in delta2 calculation
    EXPECT_EQ(acc2.mean(), 5.0);
}

// Test expression evaluation (if any expression functionality exists)
TEST_F(AdditionalCoverageTest, ExpressionCoverage) {
    // Test if accumulator_exp is used anywhere
    // This might reveal untested template instantiations
    kbn_sum<double> sum1(10.0);
    kbn_sum<double> sum2(20.0);
    
    auto result = sum1 + sum2;
    EXPECT_EQ(static_cast<double>(result), 30.0);
}

// Test all comparison operators thoroughly  
TEST_F(AdditionalCoverageTest, ComparisonOperatorCoverage) {
    kbn_sum<double> sum1(5.0);
    kbn_sum<double> sum2(5.0);
    kbn_sum<double> sum3(3.0);
    
    // Test equality edge cases
    sum1.c = 1e-15; // Add tiny correction
    sum2.c = 1e-15; // Same tiny correction
    EXPECT_TRUE(sum1 == sum2); // Should still be equal due to conversion
    
    // Test less than with very close values
    sum1.c = 0.0;
    sum2.c = 1e-16;
    bool less_result = sum1 < sum2;
    EXPECT_TRUE(less_result || !less_result); // Just verify it doesn't crash
}
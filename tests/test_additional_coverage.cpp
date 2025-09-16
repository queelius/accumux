#include <gtest/gtest.h>
#include <vector>
#include <limits>
#include "include/accumux/accumulators/kbn_sum.hpp"
#include "include/accumux/accumulators/welford.hpp"

using namespace accumux;

class AdditionalCoverageTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test specific edge cases that might not be covered
TEST_F(AdditionalCoverageTest, KBNSumSpecialCases) {
    // Test the specific branches in the += operator
    kbn_sum<double> sum;
    
    // Test edge case with values of equal magnitude
    sum = kbn_sum<double>(5.0);
    sum += -5.0;  // Should result in zero
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
    welford_accumulator<double> acc;
    
    // Test with infinity values (if supported)
    acc += std::numeric_limits<double>::max();
    acc += std::numeric_limits<double>::max();
    
    EXPECT_TRUE(std::isfinite(acc.mean()) || std::isinf(acc.mean()));
    EXPECT_EQ(acc.size(), 2u);
    
    // Test move semantics path explicitly
    welford_accumulator<double> acc2;
    double temp = 5.0;
    acc2 += std::move(temp); // This triggers the move in delta2 calculation
    EXPECT_EQ(acc2.mean(), 5.0);
}

// Test basic accumulator operations
TEST_F(AdditionalCoverageTest, BasicOperationsCoverage) {
    // Test basic accumulator arithmetic
    kbn_sum<double> sum1(10.0);
    kbn_sum<double> sum2(20.0);
    
    auto result = sum1 + sum2;
    EXPECT_EQ(static_cast<double>(result), 30.0);
    
    // Test individual accumulator functionality
    kbn_sum<double> sum;
    welford_accumulator<double> welford;
    
    sum += 42.0;
    welford += 42.0;
    
    EXPECT_EQ(static_cast<double>(sum), 42.0);
    EXPECT_EQ(welford.mean(), 42.0);
    EXPECT_EQ(welford.size(), 1u);
}

// Test all comparison operators thoroughly  
TEST_F(AdditionalCoverageTest, ComparisonOperatorCoverage) {
    kbn_sum<double> sum1(5.0);
    kbn_sum<double> sum2(5.0);
    kbn_sum<double> sum3(3.0);
    
    // Test equality
    EXPECT_TRUE(sum1 == sum2);
    EXPECT_FALSE(sum1 == sum3);
    
    // Test less than operators
    EXPECT_FALSE(sum1 < sum2);  // 5.0 < 5.0 should be false
    EXPECT_FALSE(sum1 < sum3);  // 5.0 < 3.0 should be false
    EXPECT_TRUE(sum3 < sum1);   // 3.0 < 5.0 should be true
    
    // Test less than with scalar
    EXPECT_TRUE(sum3 < 4.0);
    EXPECT_FALSE(sum1 < 4.0);
}
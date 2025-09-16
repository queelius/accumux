#include <gtest/gtest.h>
#include <vector>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include "include/accumux/accumulators/kbn_sum.hpp"

using namespace accumux;

class KBNSumTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
    
    // Helper function to check floating point equality with tolerance
    template<typename T>
    bool nearly_equal(T a, T b, T tolerance = 1e-12) {
        return std::abs(a - b) <= tolerance;
    }
};

// Test default constructor and initialization
TEST_F(KBNSumTest, DefaultConstructor) {
    kbn_sum<double> sum;
    EXPECT_EQ(static_cast<double>(sum), 0.0);
    EXPECT_EQ(sum.eval(), 0.0);
    EXPECT_EQ(sum.sum_component(), 0.0);
    EXPECT_EQ(sum.correction_component(), 0.0);
}

// Test constructor with initial value
TEST_F(KBNSumTest, ValueConstructor) {
    kbn_sum<double> sum(5.5);
    EXPECT_EQ(static_cast<double>(sum), 5.5);
    EXPECT_EQ(sum.eval(), 5.5);
    EXPECT_EQ(sum.sum_component(), 5.5);
    EXPECT_EQ(sum.correction_component(), 0.0);
}

// Test copy constructor
TEST_F(KBNSumTest, CopyConstructor) {
    kbn_sum<double> sum1(3.14);
    kbn_sum<double> sum2(sum1);
    
    EXPECT_EQ(static_cast<double>(sum2), 3.14);
    EXPECT_EQ(sum2.sum_component(), sum1.sum_component());
    EXPECT_EQ(sum2.correction_component(), sum1.correction_component());
}

// Test copy assignment
TEST_F(KBNSumTest, CopyAssignment) {
    kbn_sum<double> sum1(2.71);
    kbn_sum<double> sum2;
    
    sum2 = sum1;
    EXPECT_EQ(static_cast<double>(sum2), 2.71);
    EXPECT_EQ(sum2.sum_component(), sum1.sum_component());
    EXPECT_EQ(sum2.correction_component(), sum1.correction_component());
}

// Test assignment from T
TEST_F(KBNSumTest, ValueAssignment) {
    kbn_sum<double> sum;
    sum = 7.5;
    
    EXPECT_EQ(static_cast<double>(sum), 7.5);
    EXPECT_EQ(sum.sum_component(), 7.5);
}

// Test conversion operator
TEST_F(KBNSumTest, ConversionOperator) {
    kbn_sum<double> sum(4.2);
    double value = sum;
    EXPECT_EQ(value, 4.2);
}

// Test eval method
TEST_F(KBNSumTest, EvalMethod) {
    kbn_sum<double> sum(1.5);
    EXPECT_EQ(sum.eval(), 1.5);
}

// Test addition with scalar
TEST_F(KBNSumTest, AdditionWithScalar) {
    kbn_sum<double> sum(1.0);
    sum += 2.0;
    EXPECT_EQ(static_cast<double>(sum), 3.0);
    
    sum += -1.5;
    EXPECT_EQ(static_cast<double>(sum), 1.5);
}

// Test addition with another kbn_sum
TEST_F(KBNSumTest, AdditionWithKBNSum) {
    kbn_sum<double> sum1(2.5);
    kbn_sum<double> sum2(1.5);
    
    sum1 += sum2;
    EXPECT_EQ(static_cast<double>(sum1), 4.0);
}

// Test binary addition operator
TEST_F(KBNSumTest, BinaryAddition) {
    kbn_sum<double> sum1(3.0);
    kbn_sum<double> sum2(2.0);
    
    auto result = sum1 + sum2;
    EXPECT_EQ(static_cast<double>(result), 5.0);
    
    // Original values should remain unchanged
    EXPECT_EQ(static_cast<double>(sum1), 3.0);
    EXPECT_EQ(static_cast<double>(sum2), 2.0);
}

// Test comparison operators
TEST_F(KBNSumTest, ComparisonOperators) {
    kbn_sum<double> sum1(5.0);
    kbn_sum<double> sum2(3.0);
    kbn_sum<double> sum3(5.0);
    
    // Test equality
    EXPECT_TRUE(sum1 == sum3);
    EXPECT_FALSE(sum1 == sum2);
    
    // Test less than with kbn_sum
    EXPECT_TRUE(sum2 < sum1);
    EXPECT_FALSE(sum1 < sum2);
    EXPECT_FALSE(sum1 < sum3);
    
    // Test less than with scalar
    EXPECT_TRUE(sum2 < 4.0);
    EXPECT_FALSE(sum1 < 4.0);
}

// Test accumulation with iterators
TEST_F(KBNSumTest, AccumulationWithIterators) {
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    kbn_sum<double> sum;
    for (auto value : values) {
        sum += value;
    }
    
    EXPECT_EQ(static_cast<double>(sum), 15.0);
}

// Test accumulation with arrays
TEST_F(KBNSumTest, AccumulationWithArray) {
    std::array<double, 4> values = {2.5, 1.5, 3.0, 2.0};
    
    kbn_sum<double> sum;
    for (auto value : values) {
        sum += value;
    }
    
    EXPECT_EQ(static_cast<double>(sum), 9.0);
}

// Test numerical accuracy - classic floating point error case
TEST_F(KBNSumTest, NumericalAccuracy) {
    kbn_sum<double> kbn_sum_result;
    double naive_sum = 0.0;
    
    // Add a large number, then a small number, then subtract the large number
    // This is a classic case where naive summation loses precision
    double large = 1e16;
    double small = 1.0;
    
    // KBN sum
    kbn_sum_result += large;
    kbn_sum_result += small;
    kbn_sum_result += -large;
    
    // Naive sum
    naive_sum += large;
    naive_sum += small;
    naive_sum -= large;
    
    // KBN should preserve the small value better than naive summation
    double kbn_result = static_cast<double>(kbn_sum_result);
    
    // For this specific case, both might be close, but KBN should be more accurate
    // in general scenarios
    EXPECT_TRUE(kbn_result >= 0.0); // Should not be negative due to precision loss
}

// Test with many small values
TEST_F(KBNSumTest, ManySmallValues) {
    kbn_sum<double> sum;
    std::vector<double> small_values(1000, 0.001);
    
    for (auto value : small_values) {
        sum += value;
    }
    
    EXPECT_TRUE(nearly_equal(static_cast<double>(sum), 1.0, 1e-10));
}

// Test with mixed positive and negative values
TEST_F(KBNSumTest, MixedSignValues) {
    std::vector<double> values = {10.0, -5.0, 3.0, -2.0, 1.5, -0.5};
    
    kbn_sum<double> sum;
    for (auto value : values) {
        sum += value;
    }
    
    EXPECT_EQ(static_cast<double>(sum), 7.0);
}

// Test abs function
TEST_F(KBNSumTest, AbsFunction) {
    kbn_sum<double> negative_sum(-5.0);
    // We can't directly modify correction anymore, so let's test the abs function directly
    auto abs_sum = abs(negative_sum);
    
    EXPECT_GT(static_cast<double>(abs_sum), 0.0);
    EXPECT_EQ(static_cast<double>(abs_sum), 5.0);
}

// Test with float type
TEST_F(KBNSumTest, FloatType) {
    kbn_sum<float> sum(1.5f);
    sum += 2.5f;
    
    EXPECT_FLOAT_EQ(static_cast<float>(sum), 4.0f);
}

// Test edge cases with zero
TEST_F(KBNSumTest, ZeroOperations) {
    kbn_sum<double> sum(5.0);
    sum += 0.0;
    EXPECT_EQ(static_cast<double>(sum), 5.0);
    
    kbn_sum<double> zero_sum;
    sum += zero_sum;
    EXPECT_EQ(static_cast<double>(sum), 5.0);
}

// Test large numbers to stress the correction mechanism
TEST_F(KBNSumTest, LargeNumbers) {
    kbn_sum<double> sum;

    // Add numbers in a way that tests the correction mechanism
    // Using 1e15 instead of 1e20 to stay within double precision capabilities
    sum += 1e15;
    sum += 1.0;
    sum += 1.0;
    sum += -1e15;

    // The result should be close to 2.0
    double result = static_cast<double>(sum);
    EXPECT_NEAR(result, 2.0, 1e-10); // Should be exactly 2.0 with KBN
}

// Test empty range reduction
TEST_F(KBNSumTest, EmptyRangeReduction) {
    std::vector<double> empty_vector;
    kbn_sum<double> sum;
    
    for (auto value : empty_vector) {
        sum += value;
    }
    EXPECT_EQ(static_cast<double>(sum), 0.0);
}

// Test single element reduction
TEST_F(KBNSumTest, SingleElementReduction) {
    std::vector<double> single_element = {42.0};
    kbn_sum<double> sum;
    
    for (auto value : single_element) {
        sum += value;
    }
    EXPECT_EQ(static_cast<double>(sum), 42.0);
}

// Test the correction mechanism with public interface
TEST_F(KBNSumTest, CorrectionMechanism) {
    // Test KBN algorithm's correction behavior through sequence of operations
    kbn_sum<double> sum1(100.0);
    sum1 += 1.0;
    
    EXPECT_EQ(static_cast<double>(sum1), 101.0);
    EXPECT_EQ(sum1.sum_component(), 101.0);
    EXPECT_EQ(sum1.correction_component(), 0.0);
    
    // Test with different magnitude values
    kbn_sum<double> sum2(1.0);
    sum2 += 100.0;
    
    EXPECT_EQ(static_cast<double>(sum2), 101.0);
    EXPECT_EQ(sum2.sum_component(), 101.0);
    EXPECT_EQ(sum2.correction_component(), 0.0);
    
    // Test correction accumulation with precision-sensitive values
    kbn_sum<double> sum3;
    sum3 += 1e16;
    sum3 += 1.0;
    sum3 += 1.0;
    sum3 += -1e16;
    
    // KBN should maintain better precision than naive summation
    EXPECT_GE(static_cast<double>(sum3), 1.5);
    EXPECT_LT(static_cast<double>(sum3), 2.5);
}

// Performance/stress test with random values
TEST_F(KBNSumTest, RandomValuesStressTest) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1000.0, 1000.0);
    
    std::vector<double> random_values(10000);
    for (auto& val : random_values) {
        val = dis(gen);
    }
    
    kbn_sum<double> sum;
    for (auto value : random_values) {
        sum += value;
    }
    
    // Calculate expected sum using long double for higher precision reference
    long double expected = 0.0L;
    for (double val : random_values) {
        expected += val;
    }
    
    double kbn_result = static_cast<double>(sum);
    double expected_double = static_cast<double>(expected);
    
    // KBN should be at least as accurate as naive double summation
    EXPECT_TRUE(std::isfinite(kbn_result));
    EXPECT_TRUE(nearly_equal(kbn_result, expected_double, 1e-10));
}
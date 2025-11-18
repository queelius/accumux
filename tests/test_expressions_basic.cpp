#include <gtest/gtest.h>
#include "include/accumux/accumulators/kbn_sum.hpp"
#include "include/accumux/accumulators/welford.hpp"
#include "include/accumux/core/composition.hpp"

using namespace accumux;

class CompositionBasicTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test that composition headers compile and basic operations work
TEST_F(CompositionBasicTest, HeadersCompileSuccessfully) {
    // If this test compiles and runs, it means the composition
    // headers are syntactically correct and can be included
    kbn_sum<double> sum1(10.0);
    kbn_sum<double> sum2(20.0);

    // operator+ creates a parallel_composition, not a sum
    auto result = sum1 + sum2;
    auto [first, second] = result.eval();
    EXPECT_EQ(first, 10.0);
    EXPECT_EQ(second, 20.0);
}

// Test that basic accumulator types work
TEST_F(CompositionBasicTest, BasicAccumulatorTypes) {
    // Test that our basic types exist and work
    kbn_sum<double> sum;
    welford_accumulator<double> welford;
    
    // Add some values to each
    sum += 1.0;
    sum += 2.0;
    sum += 3.0;
    
    welford += 1.0;
    welford += 2.0;
    welford += 3.0;
    
    // Check results
    EXPECT_EQ(static_cast<double>(sum), 6.0);  // sum
    EXPECT_EQ(welford.mean(), 2.0);  // mean
    EXPECT_EQ(welford.size(), 3u);
}

// Test accumulator interoperability 
TEST_F(CompositionBasicTest, AccumulatorInteroperability) {
    // Test that accumulators can work together conceptually
    kbn_sum<double> sum;
    welford_accumulator<double> welford;
    
    // Process the same data with both
    std::vector<double> data = {1.0, 2.0, 3.0};
    
    for (auto value : data) {
        sum += value;
        welford += value; 
    }
    
    EXPECT_EQ(static_cast<double>(sum), 6.0);
    EXPECT_EQ(welford.mean(), 2.0);
    EXPECT_EQ(welford.size(), 3u);
    
    // Test feeding one result into another
    welford_accumulator<double> welford2;
    welford2 += sum.eval();  // Feed sum result into welford
    
    EXPECT_EQ(welford2.mean(), 6.0);
    EXPECT_EQ(welford2.size(), 1u);
}
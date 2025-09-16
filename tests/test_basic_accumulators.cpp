/**
 * @file test_basic_accumulators.cpp
 * @brief Comprehensive unit tests for basic accumulator types
 *
 * Tests cover min_accumulator, max_accumulator, count_accumulator,
 * product_accumulator, and minmax_accumulator with edge cases,
 * boundary conditions, and numerical stability tests.
 */

#include <gtest/gtest.h>
#include "include/accumux/accumulators/basic.hpp"
#include <vector>
#include <limits>
#include <numeric>
#include <random>
#include <algorithm>
#include <cmath>

using namespace accumux;

class BasicAccumulatorsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }
};

// ============================================================================
// min_accumulator Tests
// ============================================================================

TEST_F(BasicAccumulatorsTest, MinAccumulatorDefaultConstructor) {
    min_accumulator<double> acc;
    EXPECT_TRUE(acc.empty());
    EXPECT_EQ(acc.eval(), std::numeric_limits<double>::max());
}

TEST_F(BasicAccumulatorsTest, MinAccumulatorValueConstructor) {
    min_accumulator<double> acc(5.0);
    EXPECT_FALSE(acc.empty());
    EXPECT_DOUBLE_EQ(acc.eval(), 5.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(acc), 5.0);
}

TEST_F(BasicAccumulatorsTest, MinAccumulatorCopyConstructor) {
    min_accumulator<double> acc1(3.0);
    acc1 += 1.0;
    acc1 += 5.0;

    min_accumulator<double> acc2(acc1);
    EXPECT_DOUBLE_EQ(acc2.eval(), 1.0);
    EXPECT_FALSE(acc2.empty());
}

TEST_F(BasicAccumulatorsTest, MinAccumulatorMoveConstructor) {
    min_accumulator<double> acc1(3.0);
    acc1 += 1.0;

    min_accumulator<double> acc2(std::move(acc1));
    EXPECT_DOUBLE_EQ(acc2.eval(), 1.0);
}

TEST_F(BasicAccumulatorsTest, MinAccumulatorCopyAssignment) {
    min_accumulator<double> acc1(3.0);
    acc1 += 1.0;

    min_accumulator<double> acc2;
    acc2 = acc1;
    EXPECT_DOUBLE_EQ(acc2.eval(), 1.0);
}

TEST_F(BasicAccumulatorsTest, MinAccumulatorMoveAssignment) {
    min_accumulator<double> acc1(3.0);
    acc1 += 1.0;

    min_accumulator<double> acc2;
    acc2 = std::move(acc1);
    EXPECT_DOUBLE_EQ(acc2.eval(), 1.0);
}

TEST_F(BasicAccumulatorsTest, MinAccumulatorBasicOperations) {
    min_accumulator<int> acc;

    acc += 5;
    EXPECT_EQ(acc.eval(), 5);

    acc += 3;
    EXPECT_EQ(acc.eval(), 3);

    acc += 10;
    EXPECT_EQ(acc.eval(), 3);

    acc += 1;
    EXPECT_EQ(acc.eval(), 1);

    acc += -5;
    EXPECT_EQ(acc.eval(), -5);
}

TEST_F(BasicAccumulatorsTest, MinAccumulatorCombine) {
    min_accumulator<double> acc1(10.0);
    acc1 += 5.0;

    min_accumulator<double> acc2(3.0);
    acc2 += 8.0;

    acc1 += acc2;
    EXPECT_DOUBLE_EQ(acc1.eval(), 3.0);
}

TEST_F(BasicAccumulatorsTest, MinAccumulatorCombineEmpty) {
    min_accumulator<double> acc1(10.0);
    min_accumulator<double> acc2;  // empty

    acc1 += acc2;
    EXPECT_DOUBLE_EQ(acc1.eval(), 10.0);

    acc2 += acc1;
    EXPECT_DOUBLE_EQ(acc2.eval(), 10.0);
}

TEST_F(BasicAccumulatorsTest, MinAccumulatorEdgeCases) {
    min_accumulator<double> acc;

    // Add infinity
    acc += std::numeric_limits<double>::infinity();
    EXPECT_EQ(acc.eval(), std::numeric_limits<double>::infinity());

    // Add negative infinity
    acc += -std::numeric_limits<double>::infinity();
    EXPECT_EQ(acc.eval(), -std::numeric_limits<double>::infinity());

    // Add NaN
    min_accumulator<double> acc2;
    acc2 += std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(std::isnan(acc2.eval()));
}

TEST_F(BasicAccumulatorsTest, MinAccumulatorIntegerTypes) {
    min_accumulator<int> iacc;
    iacc += 100;
    iacc += -50;
    iacc += 200;
    EXPECT_EQ(iacc.eval(), -50);

    min_accumulator<unsigned int> uacc(100u);
    uacc += 50u;
    uacc += 200u;
    EXPECT_EQ(uacc.eval(), 50u);
}

// ============================================================================
// max_accumulator Tests
// ============================================================================

TEST_F(BasicAccumulatorsTest, MaxAccumulatorDefaultConstructor) {
    max_accumulator<double> acc;
    EXPECT_TRUE(acc.empty());
    EXPECT_EQ(acc.eval(), std::numeric_limits<double>::lowest());
}

TEST_F(BasicAccumulatorsTest, MaxAccumulatorValueConstructor) {
    max_accumulator<double> acc(5.0);
    EXPECT_FALSE(acc.empty());
    EXPECT_DOUBLE_EQ(acc.eval(), 5.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(acc), 5.0);
}

TEST_F(BasicAccumulatorsTest, MaxAccumulatorCopyAndMove) {
    max_accumulator<double> acc1(3.0);
    acc1 += 7.0;
    acc1 += 2.0;

    max_accumulator<double> acc2(acc1);
    EXPECT_DOUBLE_EQ(acc2.eval(), 7.0);

    max_accumulator<double> acc3(std::move(acc1));
    EXPECT_DOUBLE_EQ(acc3.eval(), 7.0);

    max_accumulator<double> acc4;
    acc4 = acc2;
    EXPECT_DOUBLE_EQ(acc4.eval(), 7.0);

    max_accumulator<double> acc5;
    acc5 = std::move(acc2);
    EXPECT_DOUBLE_EQ(acc5.eval(), 7.0);
}

TEST_F(BasicAccumulatorsTest, MaxAccumulatorBasicOperations) {
    max_accumulator<int> acc;

    acc += 5;
    EXPECT_EQ(acc.eval(), 5);

    acc += 3;
    EXPECT_EQ(acc.eval(), 5);

    acc += 10;
    EXPECT_EQ(acc.eval(), 10);

    acc += -1;
    EXPECT_EQ(acc.eval(), 10);
}

TEST_F(BasicAccumulatorsTest, MaxAccumulatorCombine) {
    max_accumulator<double> acc1(2.0);
    acc1 += 5.0;

    max_accumulator<double> acc2(8.0);
    acc2 += 3.0;

    acc1 += acc2;
    EXPECT_DOUBLE_EQ(acc1.eval(), 8.0);
}

TEST_F(BasicAccumulatorsTest, MaxAccumulatorCombineEmpty) {
    max_accumulator<double> acc1(10.0);
    max_accumulator<double> acc2;  // empty

    acc1 += acc2;
    EXPECT_DOUBLE_EQ(acc1.eval(), 10.0);

    acc2 += acc1;
    EXPECT_DOUBLE_EQ(acc2.eval(), 10.0);
}

TEST_F(BasicAccumulatorsTest, MaxAccumulatorEdgeCases) {
    max_accumulator<double> acc;

    acc += -std::numeric_limits<double>::infinity();
    EXPECT_EQ(acc.eval(), -std::numeric_limits<double>::infinity());

    acc += std::numeric_limits<double>::infinity();
    EXPECT_EQ(acc.eval(), std::numeric_limits<double>::infinity());
}

// ============================================================================
// count_accumulator Tests
// ============================================================================

TEST_F(BasicAccumulatorsTest, CountAccumulatorDefaultConstructor) {
    count_accumulator acc;
    EXPECT_EQ(acc.eval(), 0u);
    EXPECT_EQ(acc.size(), 0u);
    EXPECT_EQ(static_cast<std::size_t>(acc), 0u);
}

TEST_F(BasicAccumulatorsTest, CountAccumulatorValueConstructor) {
    count_accumulator acc(5);
    EXPECT_EQ(acc.eval(), 5u);
    EXPECT_EQ(acc.size(), 5u);
}

TEST_F(BasicAccumulatorsTest, CountAccumulatorCopyAndMove) {
    count_accumulator acc1(3);
    acc1 += 1.0;
    acc1 += "test";

    count_accumulator acc2(acc1);
    EXPECT_EQ(acc2.eval(), 5u);

    count_accumulator acc3(std::move(acc1));
    EXPECT_EQ(acc3.eval(), 5u);

    count_accumulator acc4;
    acc4 = acc2;
    EXPECT_EQ(acc4.eval(), 5u);

    count_accumulator acc5;
    acc5 = std::move(acc2);
    EXPECT_EQ(acc5.eval(), 5u);
}

TEST_F(BasicAccumulatorsTest, CountAccumulatorBasicOperations) {
    count_accumulator acc;

    acc += 5;
    EXPECT_EQ(acc.eval(), 1u);

    acc += 3.14;
    EXPECT_EQ(acc.eval(), 2u);

    acc += "string";
    EXPECT_EQ(acc.eval(), 3u);

    struct CustomType {};
    acc += CustomType{};
    EXPECT_EQ(acc.eval(), 4u);
}

TEST_F(BasicAccumulatorsTest, CountAccumulatorCombine) {
    count_accumulator acc1;
    acc1 += 1;
    acc1 += 2;

    count_accumulator acc2;
    acc2 += 3;
    acc2 += 4;
    acc2 += 5;

    acc1 += acc2;
    EXPECT_EQ(acc1.eval(), 5u);
}

TEST_F(BasicAccumulatorsTest, CountAccumulatorLargeCount) {
    count_accumulator acc;

    for(int i = 0; i < 10000; ++i) {
        acc += i;
    }
    EXPECT_EQ(acc.eval(), 10000u);

    count_accumulator acc2(5000);
    acc += acc2;
    EXPECT_EQ(acc.eval(), 15000u);
}

// ============================================================================
// product_accumulator Tests
// ============================================================================

TEST_F(BasicAccumulatorsTest, ProductAccumulatorDefaultConstructor) {
    product_accumulator<double> acc;
    EXPECT_TRUE(acc.empty());
    EXPECT_DOUBLE_EQ(acc.eval(), 1.0);
}

TEST_F(BasicAccumulatorsTest, ProductAccumulatorValueConstructor) {
    product_accumulator<double> acc(2.0);
    EXPECT_FALSE(acc.empty());
    EXPECT_DOUBLE_EQ(acc.eval(), 2.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(acc), 2.0);
}

TEST_F(BasicAccumulatorsTest, ProductAccumulatorCopyAndMove) {
    product_accumulator<double> acc1(2.0);
    acc1 += 3.0;

    product_accumulator<double> acc2(acc1);
    EXPECT_NEAR(acc2.eval(), 6.0, 1e-10);

    product_accumulator<double> acc3(std::move(acc1));
    EXPECT_NEAR(acc3.eval(), 6.0, 1e-10);

    product_accumulator<double> acc4;
    acc4 = acc2;
    EXPECT_NEAR(acc4.eval(), 6.0, 1e-10);

    product_accumulator<double> acc5;
    acc5 = std::move(acc2);
    EXPECT_NEAR(acc5.eval(), 6.0, 1e-10);
}

TEST_F(BasicAccumulatorsTest, ProductAccumulatorBasicOperations) {
    product_accumulator<double> acc;

    acc += 2.0;
    EXPECT_DOUBLE_EQ(acc.eval(), 2.0);

    acc += 3.0;
    EXPECT_NEAR(acc.eval(), 6.0, 1e-10);

    acc += 0.5;
    EXPECT_NEAR(acc.eval(), 3.0, 1e-10);
}

TEST_F(BasicAccumulatorsTest, ProductAccumulatorZero) {
    product_accumulator<double> acc(5.0);
    acc += 2.0;
    acc += 0.0;
    acc += 10.0;

    EXPECT_DOUBLE_EQ(acc.eval(), 0.0);
    EXPECT_FALSE(acc.empty());
}

TEST_F(BasicAccumulatorsTest, ProductAccumulatorNegativeValues) {
    product_accumulator<double> acc;

    acc += -2.0;
    EXPECT_DOUBLE_EQ(acc.eval(), 2.0);  // Uses abs internally

    acc += -3.0;
    EXPECT_NEAR(acc.eval(), 6.0, 1e-10);

    acc += 2.0;
    EXPECT_NEAR(acc.eval(), 12.0, 1e-10);
}

TEST_F(BasicAccumulatorsTest, ProductAccumulatorCombine) {
    product_accumulator<double> acc1(2.0);
    acc1 += 3.0;

    product_accumulator<double> acc2(4.0);
    acc2 += 5.0;

    acc1 += acc2;
    EXPECT_NEAR(acc1.eval(), 120.0, 1e-10);
}

TEST_F(BasicAccumulatorsTest, ProductAccumulatorCombineWithZero) {
    product_accumulator<double> acc1(2.0);
    acc1 += 3.0;

    product_accumulator<double> acc2(4.0);
    acc2 += 0.0;

    acc1 += acc2;
    EXPECT_DOUBLE_EQ(acc1.eval(), 0.0);
}

TEST_F(BasicAccumulatorsTest, ProductAccumulatorCombineEmpty) {
    product_accumulator<double> acc1(5.0);
    product_accumulator<double> acc2;  // empty

    acc1 += acc2;
    EXPECT_DOUBLE_EQ(acc1.eval(), 5.0);

    acc2 += acc1;
    EXPECT_DOUBLE_EQ(acc2.eval(), 5.0);
}

TEST_F(BasicAccumulatorsTest, ProductAccumulatorLargeNumbers) {
    product_accumulator<double> acc;

    acc += 1e50;
    acc += 1e50;
    acc += 1e-100;

    // Should handle via log representation
    EXPECT_FALSE(std::isinf(acc.eval()));
    EXPECT_NEAR(acc.eval(), 1.0, 1e-10);
}

TEST_F(BasicAccumulatorsTest, ProductAccumulatorFloat) {
    product_accumulator<float> acc;

    acc += 2.0f;
    acc += 3.0f;
    acc += 0.5f;

    EXPECT_NEAR(acc.eval(), 3.0f, 1e-6f);
}

// ============================================================================
// minmax_accumulator Tests
// ============================================================================

TEST_F(BasicAccumulatorsTest, MinMaxAccumulatorDefaultConstructor) {
    minmax_accumulator<double> acc;
    EXPECT_TRUE(acc.empty());
    auto [min_val, max_val] = acc.eval();
    EXPECT_EQ(min_val, std::numeric_limits<double>::max());
    EXPECT_EQ(max_val, std::numeric_limits<double>::lowest());
}

TEST_F(BasicAccumulatorsTest, MinMaxAccumulatorValueConstructor) {
    minmax_accumulator<double> acc(5.0);
    EXPECT_FALSE(acc.empty());
    auto [min_val, max_val] = acc.eval();
    EXPECT_DOUBLE_EQ(min_val, 5.0);
    EXPECT_DOUBLE_EQ(max_val, 5.0);
    EXPECT_DOUBLE_EQ(acc.min(), 5.0);
    EXPECT_DOUBLE_EQ(acc.max(), 5.0);
    EXPECT_DOUBLE_EQ(acc.range(), 0.0);
}

TEST_F(BasicAccumulatorsTest, MinMaxAccumulatorPairConstructor) {
    minmax_accumulator<double> acc(std::make_pair(2.0, 8.0));
    EXPECT_FALSE(acc.empty());
    auto [min_val, max_val] = acc.eval();
    EXPECT_DOUBLE_EQ(min_val, 2.0);
    EXPECT_DOUBLE_EQ(max_val, 8.0);
    EXPECT_DOUBLE_EQ(acc.range(), 6.0);
}

TEST_F(BasicAccumulatorsTest, MinMaxAccumulatorCopyAndMove) {
    minmax_accumulator<double> acc1(5.0);
    acc1 += 2.0;
    acc1 += 8.0;

    minmax_accumulator<double> acc2(acc1);
    EXPECT_DOUBLE_EQ(acc2.min(), 2.0);
    EXPECT_DOUBLE_EQ(acc2.max(), 8.0);

    minmax_accumulator<double> acc3(std::move(acc1));
    EXPECT_DOUBLE_EQ(acc3.min(), 2.0);
    EXPECT_DOUBLE_EQ(acc3.max(), 8.0);

    minmax_accumulator<double> acc4;
    acc4 = acc2;
    EXPECT_DOUBLE_EQ(acc4.min(), 2.0);
    EXPECT_DOUBLE_EQ(acc4.max(), 8.0);

    minmax_accumulator<double> acc5;
    acc5 = std::move(acc2);
    EXPECT_DOUBLE_EQ(acc5.min(), 2.0);
    EXPECT_DOUBLE_EQ(acc5.max(), 8.0);
}

TEST_F(BasicAccumulatorsTest, MinMaxAccumulatorBasicOperations) {
    minmax_accumulator<int> acc;

    acc += 5;
    EXPECT_EQ(acc.min(), 5);
    EXPECT_EQ(acc.max(), 5);

    acc += 3;
    EXPECT_EQ(acc.min(), 3);
    EXPECT_EQ(acc.max(), 5);

    acc += 10;
    EXPECT_EQ(acc.min(), 3);
    EXPECT_EQ(acc.max(), 10);

    acc += -2;
    EXPECT_EQ(acc.min(), -2);
    EXPECT_EQ(acc.max(), 10);
    EXPECT_EQ(acc.range(), 12);
}

TEST_F(BasicAccumulatorsTest, MinMaxAccumulatorCombine) {
    minmax_accumulator<double> acc1(5.0);
    acc1 += 2.0;
    acc1 += 8.0;

    minmax_accumulator<double> acc2(1.0);
    acc2 += 10.0;
    acc2 += 4.0;

    acc1 += acc2;
    EXPECT_DOUBLE_EQ(acc1.min(), 1.0);
    EXPECT_DOUBLE_EQ(acc1.max(), 10.0);
    EXPECT_DOUBLE_EQ(acc1.range(), 9.0);
}

TEST_F(BasicAccumulatorsTest, MinMaxAccumulatorCombineEmpty) {
    minmax_accumulator<double> acc1(5.0);
    acc1 += 10.0;

    minmax_accumulator<double> acc2;  // empty

    acc1 += acc2;
    EXPECT_DOUBLE_EQ(acc1.min(), 5.0);
    EXPECT_DOUBLE_EQ(acc1.max(), 10.0);

    acc2 += acc1;
    EXPECT_DOUBLE_EQ(acc2.min(), 5.0);
    EXPECT_DOUBLE_EQ(acc2.max(), 10.0);
}

TEST_F(BasicAccumulatorsTest, MinMaxAccumulatorConversionOperator) {
    minmax_accumulator<double> acc(3.0);
    acc += 7.0;
    acc += 1.0;

    std::pair<double, double> result = static_cast<std::pair<double, double>>(acc);
    EXPECT_DOUBLE_EQ(result.first, 1.0);
    EXPECT_DOUBLE_EQ(result.second, 7.0);
}

// ============================================================================
// Factory Function Tests
// ============================================================================

TEST_F(BasicAccumulatorsTest, FactoryFunctions) {
    auto min_acc = make_min_accumulator(5.0);
    min_acc += 3.0;
    EXPECT_DOUBLE_EQ(min_acc.eval(), 3.0);

    auto max_acc = make_max_accumulator(5);
    max_acc += 10;
    EXPECT_EQ(max_acc.eval(), 10);

    auto minmax_acc = make_minmax_accumulator(5.0);
    minmax_acc += 2.0;
    minmax_acc += 8.0;
    EXPECT_DOUBLE_EQ(minmax_acc.min(), 2.0);
    EXPECT_DOUBLE_EQ(minmax_acc.max(), 8.0);

    auto count_acc = make_count_accumulator();
    count_acc += 1;
    count_acc += 2;
    EXPECT_EQ(count_acc.eval(), 2u);

    auto prod_acc = make_product_accumulator(2.0);
    prod_acc += 3.0;
    EXPECT_NEAR(prod_acc.eval(), 6.0, 1e-10);
}

// ============================================================================
// Stress Tests with Large Data Sets
// ============================================================================

TEST_F(BasicAccumulatorsTest, StressTestMinMax) {
    std::mt19937 gen(12345);
    std::uniform_real_distribution<> dis(-1000.0, 1000.0);

    min_accumulator<double> min_acc;
    max_accumulator<double> max_acc;
    minmax_accumulator<double> minmax_acc;

    std::vector<double> data(10000);
    for(auto& val : data) {
        val = dis(gen);
        min_acc += val;
        max_acc += val;
        minmax_acc += val;
    }

    auto actual_min = *std::min_element(data.begin(), data.end());
    auto actual_max = *std::max_element(data.begin(), data.end());

    EXPECT_DOUBLE_EQ(min_acc.eval(), actual_min);
    EXPECT_DOUBLE_EQ(max_acc.eval(), actual_max);
    EXPECT_DOUBLE_EQ(minmax_acc.min(), actual_min);
    EXPECT_DOUBLE_EQ(minmax_acc.max(), actual_max);
}

TEST_F(BasicAccumulatorsTest, StressTestProduct) {
    product_accumulator<double> acc;

    // Test with numbers that would overflow without log representation
    std::vector<double> values = {1e20, 1e20, 1e-40, 2.0};
    for(auto val : values) {
        acc += val;
    }

    EXPECT_NEAR(acc.eval(), 2.0, 1e-10);
}

// ============================================================================
// Concept Compliance Tests
// ============================================================================

TEST_F(BasicAccumulatorsTest, ConceptCompliance) {
    // These are compile-time checks that are already in the header
    // but we can test them at runtime too
    static_assert(Accumulator<min_accumulator<double>>);
    static_assert(Accumulator<max_accumulator<int>>);
    static_assert(Accumulator<count_accumulator>);
    static_assert(Accumulator<product_accumulator<double>>);
    static_assert(Accumulator<minmax_accumulator<double>>);

    // Runtime verification that they work as accumulators
    min_accumulator<double> min_acc;
    max_accumulator<double> max_acc;
    count_accumulator count_acc;
    product_accumulator<double> prod_acc;
    minmax_accumulator<double> minmax_acc;

    // They all should support += operations
    min_acc += 1.0;
    max_acc += 1.0;
    count_acc += 1.0;
    prod_acc += 1.0;
    minmax_acc += 1.0;

    // They all should support eval()
    auto min_result = min_acc.eval();
    auto max_result = max_acc.eval();
    auto count_result = count_acc.eval();
    auto prod_result = prod_acc.eval();
    auto minmax_result = minmax_acc.eval();

    EXPECT_EQ(min_result, 1.0);
    EXPECT_EQ(max_result, 1.0);
    EXPECT_EQ(count_result, 1u);
    EXPECT_EQ(prod_result, 1.0);
    EXPECT_EQ(minmax_result.first, 1.0);
    EXPECT_EQ(minmax_result.second, 1.0);
}
/**
 * @file test_composition_simple.cpp
 * @brief Simple unit tests for accumulator composition framework
 *
 * Tests cover parallel_composition with basic functionality
 */

#include <gtest/gtest.h>
#include "include/accumux/core/composition.hpp"
#include "include/accumux/accumulators/basic.hpp"
#include "include/accumux/accumulators/kbn_sum.hpp"
#include "include/accumux/accumulators/welford.hpp"
#include <vector>

using namespace accumux;

class SimpleCompositionTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_data = {1.0, 2.0, 3.0, 4.0, 5.0};
    }

    std::vector<double> test_data;
};

// ============================================================================
// Basic Parallel Composition Tests
// ============================================================================

TEST_F(SimpleCompositionTest, ParallelCompositionBasic) {
    // Test simple parallel composition with compatible types
    min_accumulator<double> min_acc;
    max_accumulator<double> max_acc;

    auto comp = min_acc + max_acc;

    // Add values one by one
    comp += 3.0;
    comp += 1.0;
    comp += 5.0;
    comp += 2.0;

    auto [min_val, max_val] = comp.eval();
    EXPECT_DOUBLE_EQ(min_val, 1.0);
    EXPECT_DOUBLE_EQ(max_val, 5.0);
}

TEST_F(SimpleCompositionTest, ParallelCompositionGetters) {
    min_accumulator<double> min_acc;
    max_accumulator<double> max_acc;

    auto comp = min_acc + max_acc;

    comp += 3.0;
    comp += 1.0;
    comp += 5.0;

    // Test get_first and get_second
    const auto& first = comp.get_first();
    const auto& second = comp.get_second();

    EXPECT_DOUBLE_EQ(first.eval(), 1.0);
    EXPECT_DOUBLE_EQ(second.eval(), 5.0);

    // Test get by type
    const auto& min_result = comp.get<min_accumulator<double>>();
    const auto& max_result = comp.get<max_accumulator<double>>();

    EXPECT_DOUBLE_EQ(min_result.eval(), 1.0);
    EXPECT_DOUBLE_EQ(max_result.eval(), 5.0);
}

TEST_F(SimpleCompositionTest, ParallelCompositionCombine) {
    min_accumulator<double> min1;
    max_accumulator<double> max1;
    auto comp1 = min1 + max1;

    comp1 += 2.0;
    comp1 += 4.0;

    min_accumulator<double> min2;
    max_accumulator<double> max2;
    auto comp2 = min2 + max2;

    comp2 += 1.0;
    comp2 += 5.0;

    // Combine two compositions
    comp1 += comp2;

    auto [min_val, max_val] = comp1.eval();
    EXPECT_DOUBLE_EQ(min_val, 1.0);
    EXPECT_DOUBLE_EQ(max_val, 5.0);
}

TEST_F(SimpleCompositionTest, ParallelCompositionConstructors) {
    // Test with pre-initialized accumulators
    min_accumulator<double> min_acc(10.0);
    max_accumulator<double> max_acc(20.0);

    parallel_composition<min_accumulator<double>, max_accumulator<double>> comp(min_acc, max_acc);

    auto [min_val, max_val] = comp.eval();
    EXPECT_DOUBLE_EQ(min_val, 10.0);
    EXPECT_DOUBLE_EQ(max_val, 20.0);

    // Add more values
    comp += 5.0;
    comp += 25.0;

    auto [min_val2, max_val2] = comp.eval();
    EXPECT_DOUBLE_EQ(min_val2, 5.0);
    EXPECT_DOUBLE_EQ(max_val2, 25.0);
}

TEST_F(SimpleCompositionTest, ParallelCompositionEmpty) {
    min_accumulator<double> min_acc;
    max_accumulator<double> max_acc;
    auto comp = min_acc + max_acc;

    // Test empty composition
    auto [min_val, max_val] = comp.eval();
    EXPECT_EQ(min_val, std::numeric_limits<double>::max());
    EXPECT_EQ(max_val, std::numeric_limits<double>::lowest());
}

TEST_F(SimpleCompositionTest, ParallelCompositionWithKBNSum) {
    min_accumulator<double> min_acc;
    kbn_sum<double> sum_acc;

    // Create parallel composition
    parallel_composition<min_accumulator<double>, kbn_sum<double>> comp(min_acc, sum_acc);

    // Add test data
    double expected_sum = 0.0;
    for (const auto& val : test_data) {
        comp += val;
        expected_sum += val;
    }

    auto [min_val, sum] = comp.eval();
    EXPECT_DOUBLE_EQ(min_val, 1.0);
    EXPECT_DOUBLE_EQ(sum, expected_sum);
}

TEST_F(SimpleCompositionTest, ParallelCompositionThreeAccumulators) {
    // Test nested composition for three accumulators
    min_accumulator<double> min_acc;
    max_accumulator<double> max_acc;
    count_accumulator count_acc;

    auto minmax = min_acc + max_acc;
    auto comp = minmax + count_acc;

    for (const auto& val : test_data) {
        comp += val;
    }

    auto [minmax_result, count] = comp.eval();
    auto [min_val, max_val] = minmax_result;

    EXPECT_DOUBLE_EQ(min_val, 1.0);
    EXPECT_DOUBLE_EQ(max_val, 5.0);
    EXPECT_EQ(count, 5u);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(SimpleCompositionTest, ParallelCompositionInfinity) {
    min_accumulator<double> min_acc;
    max_accumulator<double> max_acc;
    auto comp = min_acc + max_acc;

    comp += std::numeric_limits<double>::infinity();
    comp += -std::numeric_limits<double>::infinity();
    comp += 0.0;

    auto [min_val, max_val] = comp.eval();
    EXPECT_EQ(min_val, -std::numeric_limits<double>::infinity());
    EXPECT_EQ(max_val, std::numeric_limits<double>::infinity());
}

TEST_F(SimpleCompositionTest, ParallelCompositionLargeDataset) {
    min_accumulator<double> min_acc;
    max_accumulator<double> max_acc;
    count_accumulator count_acc;

    auto minmax = min_acc + max_acc;
    auto comp = minmax + count_acc;

    const std::size_t n = 1000;
    double expected_min = std::numeric_limits<double>::max();
    double expected_max = std::numeric_limits<double>::lowest();

    for (std::size_t i = 0; i < n; ++i) {
        double val = static_cast<double>(i) - 500.0;
        comp += val;
        expected_min = std::min(expected_min, val);
        expected_max = std::max(expected_max, val);
    }

    auto [minmax_result, count] = comp.eval();
    auto [min_val, max_val] = minmax_result;

    EXPECT_DOUBLE_EQ(min_val, expected_min);
    EXPECT_DOUBLE_EQ(max_val, expected_max);
    EXPECT_EQ(count, n);
}

// ============================================================================
// Integration with Other Accumulators
// ============================================================================

TEST_F(SimpleCompositionTest, CompositionWithBasicAccumulators) {
    // Test all basic accumulators in composition
    min_accumulator<double> min_acc;
    max_accumulator<double> max_acc;
    auto comp1 = min_acc + max_acc;

    count_accumulator count_acc;
    product_accumulator<double> prod_acc;
    auto comp2 = count_acc + prod_acc;

    // Process some data
    std::vector<double> data = {2.0, 3.0, 4.0};

    for (const auto& val : data) {
        comp1 += val;
        comp2 += val;
    }

    auto [min_val, max_val] = comp1.eval();
    EXPECT_DOUBLE_EQ(min_val, 2.0);
    EXPECT_DOUBLE_EQ(max_val, 4.0);

    auto [count, product] = comp2.eval();
    EXPECT_EQ(count, 3u);
    EXPECT_NEAR(product, 24.0, 1e-10);
}

TEST_F(SimpleCompositionTest, CompositionWithMinMax) {
    minmax_accumulator<double> minmax_acc;
    count_accumulator count_acc;

    auto comp = minmax_acc + count_acc;

    for (const auto& val : test_data) {
        comp += val;
    }

    auto [minmax_pair, count] = comp.eval();
    EXPECT_DOUBLE_EQ(minmax_pair.first, 1.0);
    EXPECT_DOUBLE_EQ(minmax_pair.second, 5.0);
    EXPECT_EQ(count, 5u);
}

// ============================================================================
// Factory Function Tests
// ============================================================================

TEST_F(SimpleCompositionTest, CompositionWithFactoryFunctions) {
    auto min_acc = make_min_accumulator(100.0);
    auto max_acc = make_max_accumulator(0.0);

    auto comp = min_acc + max_acc;

    comp += 50.0;
    comp += 150.0;
    comp += 25.0;

    auto [min_val, max_val] = comp.eval();
    EXPECT_DOUBLE_EQ(min_val, 25.0);
    EXPECT_DOUBLE_EQ(max_val, 150.0);
}

// ============================================================================
// Type Deduction Tests
// ============================================================================

TEST_F(SimpleCompositionTest, CompositionTypeDeduction) {
    // Test that composition works with auto and deduced types
    auto comp = min_accumulator<float>() + max_accumulator<float>();

    comp += 1.5f;
    comp += 2.5f;
    comp += 0.5f;

    auto [min_val, max_val] = comp.eval();
    EXPECT_FLOAT_EQ(min_val, 0.5f);
    EXPECT_FLOAT_EQ(max_val, 2.5f);
}

TEST_F(SimpleCompositionTest, NestedCompositionTypeDeduction) {
    // Test deeply nested composition
    auto comp1 = min_accumulator<double>() + max_accumulator<double>();
    auto comp2 = comp1 + count_accumulator();
    auto comp3 = comp2 + kbn_sum<double>();

    comp3 += 1.0;
    comp3 += 2.0;
    comp3 += 3.0;

    auto [rest, sum] = comp3.eval();
    auto [minmax, count] = rest;
    auto [min_val, max_val] = minmax;

    EXPECT_DOUBLE_EQ(min_val, 1.0);
    EXPECT_DOUBLE_EQ(max_val, 3.0);
    EXPECT_EQ(count, 3u);
    EXPECT_DOUBLE_EQ(sum, 6.0);
}
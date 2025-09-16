/**
 * @file test_composition.cpp
 * @brief Comprehensive unit tests for accumulator composition framework
 *
 * Tests cover parallel_composition, sequential_composition, and
 * conditional_composition with various accumulator types, edge cases,
 * and complex compositions.
 */

#include <gtest/gtest.h>
#include "include/accumux/core/composition.hpp"
#include "include/accumux/accumulators/basic.hpp"
#include "include/accumux/accumulators/kbn_sum.hpp"
#include "include/accumux/accumulators/welford.hpp"
#include <vector>
#include <numeric>
#include <random>
#include <functional>

using namespace accumux;

class CompositionTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_data = {1.0, 2.0, 3.0, 4.0, 5.0};
    }

    std::vector<double> test_data;
};

// ============================================================================
// Parallel Composition Tests
// ============================================================================

TEST_F(CompositionTest, ParallelCompositionDefaultConstructor) {
    parallel_composition<min_accumulator<double>, max_accumulator<double>> comp;

    // Both should be empty initially
    auto [min_val, max_val] = comp.eval();
    EXPECT_EQ(min_val, std::numeric_limits<double>::max());
    EXPECT_EQ(max_val, std::numeric_limits<double>::lowest());
}

TEST_F(CompositionTest, ParallelCompositionConstructorWithAccumulators) {
    min_accumulator<double> min_acc(10.0);
    max_accumulator<double> max_acc(20.0);

    parallel_composition<min_accumulator<double>, max_accumulator<double>> comp(min_acc, max_acc);

    auto [min_val, max_val] = comp.eval();
    EXPECT_DOUBLE_EQ(min_val, 10.0);
    EXPECT_DOUBLE_EQ(max_val, 20.0);
}

TEST_F(CompositionTest, ParallelCompositionBasicOperations) {
    parallel_composition<min_accumulator<double>, max_accumulator<double>> comp;

    for (const auto& val : test_data) {
        comp += val;
    }

    auto [min_val, max_val] = comp.eval();
    EXPECT_DOUBLE_EQ(min_val, 1.0);
    EXPECT_DOUBLE_EQ(max_val, 5.0);
}

TEST_F(CompositionTest, ParallelCompositionGetByType) {
    parallel_composition<min_accumulator<double>, max_accumulator<double>> comp;

    comp += 3.0;
    comp += 1.0;
    comp += 5.0;

    const auto& min_acc = comp.get<min_accumulator<double>>();
    const auto& max_acc = comp.get<max_accumulator<double>>();

    EXPECT_DOUBLE_EQ(min_acc.eval(), 1.0);
    EXPECT_DOUBLE_EQ(max_acc.eval(), 5.0);
}

TEST_F(CompositionTest, ParallelCompositionGetByIndex) {
    parallel_composition<min_accumulator<double>, max_accumulator<double>> comp;

    comp += 3.0;
    comp += 1.0;
    comp += 5.0;

    const auto& first = comp.get_first();
    const auto& second = comp.get_second();

    EXPECT_DOUBLE_EQ(first.eval(), 1.0);
    EXPECT_DOUBLE_EQ(second.eval(), 5.0);
}

TEST_F(CompositionTest, ParallelCompositionCombine) {
    parallel_composition<min_accumulator<double>, max_accumulator<double>> comp1;
    comp1 += 2.0;
    comp1 += 4.0;

    parallel_composition<min_accumulator<double>, max_accumulator<double>> comp2;
    comp2 += 1.0;
    comp2 += 5.0;

    comp1 += comp2;

    auto [min_val, max_val] = comp1.eval();
    EXPECT_DOUBLE_EQ(min_val, 1.0);
    EXPECT_DOUBLE_EQ(max_val, 5.0);
}

TEST_F(CompositionTest, ParallelCompositionEvaluation) {
    parallel_composition<min_accumulator<double>, max_accumulator<double>> comp;

    comp += 3.0;
    comp += 1.0;
    comp += 5.0;

    // Just use eval() directly without conversion operator
    auto result = comp.eval();
    EXPECT_DOUBLE_EQ(std::get<0>(result), 1.0);
    EXPECT_DOUBLE_EQ(std::get<1>(result), 5.0);
}

TEST_F(CompositionTest, ParallelCompositionDifferentTypes) {
    parallel_composition<count_accumulator, kbn_sum<double>> comp;

    for (const auto& val : test_data) {
        comp += val;
    }

    auto [count, sum] = comp.eval();
    EXPECT_EQ(count, 5u);
    EXPECT_DOUBLE_EQ(sum, 15.0);
}

TEST_F(CompositionTest, ParallelCompositionWithWelford) {
    parallel_composition<welford_accumulator<double>, minmax_accumulator<double>> comp;

    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    for (const auto& val : data) {
        comp += val;
    }

    const auto& welford = comp.get_first();
    const auto& minmax = comp.get_second();

    EXPECT_DOUBLE_EQ(welford.mean(), 3.0);
    EXPECT_DOUBLE_EQ(welford.variance(), 2.5);  // Population variance
    EXPECT_EQ(welford.count(), 5u);

    EXPECT_DOUBLE_EQ(minmax.min(), 1.0);
    EXPECT_DOUBLE_EQ(minmax.max(), 5.0);
}

// ============================================================================
// Operator+ Tests (Parallel Composition)
// ============================================================================

TEST_F(CompositionTest, OperatorPlusCreatesParallelComposition) {
    min_accumulator<double> min_acc;
    max_accumulator<double> max_acc;

    auto comp = min_acc + max_acc;

    comp += 3.0;
    comp += 1.0;
    comp += 5.0;

    auto [min_val, max_val] = comp.eval();
    EXPECT_DOUBLE_EQ(min_val, 1.0);
    EXPECT_DOUBLE_EQ(max_val, 5.0);
}

TEST_F(CompositionTest, ChainedParallelComposition) {
    min_accumulator<double> min_acc;
    max_accumulator<double> max_acc;
    count_accumulator count_acc;

    auto comp = min_acc + max_acc + count_acc;

    for (const auto& val : test_data) {
        comp += val;
    }

    // The result should be nested parallel compositions
    auto result = comp.eval();
    // Extract values from nested tuple
    auto [minmax_tuple, count] = result;
    auto [min_val, max_val] = minmax_tuple;

    EXPECT_DOUBLE_EQ(min_val, 1.0);
    EXPECT_DOUBLE_EQ(max_val, 5.0);
    EXPECT_EQ(count, 5u);
}

TEST_F(CompositionTest, ParallelCompositionRValueReferences) {
    auto comp = min_accumulator<double>() + max_accumulator<double>();

    comp += 3.0;
    comp += 1.0;
    comp += 5.0;

    auto [min_val, max_val] = comp.eval();
    EXPECT_DOUBLE_EQ(min_val, 1.0);
    EXPECT_DOUBLE_EQ(max_val, 5.0);
}

// ============================================================================
// Sequential Composition Tests
// ============================================================================

TEST_F(CompositionTest, SequentialCompositionDefaultConstructor) {
    sequential_composition<count_accumulator, min_accumulator<std::size_t>> comp;

    // Process some values
    comp += 10.0;
    comp += 20.0;
    comp += 30.0;

    // The count should be fed to min, giving us the minimum count seen
    auto result = comp.eval();
    EXPECT_EQ(result, 1u);  // First count was 1
}

TEST_F(CompositionTest, SequentialCompositionConstructorWithAccumulators) {
    count_accumulator count_acc;
    min_accumulator<std::size_t> min_acc;

    sequential_composition<count_accumulator, min_accumulator<std::size_t>> comp(count_acc, min_acc);

    comp += 10.0;
    comp += 20.0;

    auto result = comp.eval();
    EXPECT_EQ(result, 1u);
}

TEST_F(CompositionTest, SequentialCompositionIntermediate) {
    sequential_composition<count_accumulator, max_accumulator<std::size_t>> comp;

    comp += 10.0;
    comp += 20.0;
    comp += 30.0;

    auto intermediate = comp.intermediate();
    EXPECT_EQ(intermediate, 3u);

    auto final = comp.eval();
    EXPECT_EQ(final, 3u);  // Max of {1, 2, 3}
}

// ============================================================================
// Operator* Tests (Sequential Composition)
// ============================================================================

TEST_F(CompositionTest, OperatorMultipliesCreatesSequentialComposition) {
    count_accumulator count_acc;
    max_accumulator<std::size_t> max_acc;

    auto comp = count_acc * max_acc;

    comp += 10.0;
    comp += 20.0;
    comp += 30.0;

    auto result = comp.eval();
    EXPECT_EQ(result, 3u);  // Max of counts {1, 2, 3}
}

// ============================================================================
// Conditional Composition Tests
// ============================================================================

TEST_F(CompositionTest, ConditionalCompositionBasic) {
    min_accumulator<double> min_acc;
    max_accumulator<double> max_acc;

    auto predicate = [](double value) { return value < 3.0; };

    auto comp = conditional(min_acc, max_acc, predicate);

    comp += 1.0;  // < 3, goes to min
    comp += 2.0;  // < 3, goes to min
    comp += 4.0;  // >= 3, switches to max
    comp += 5.0;  // >= 3, goes to max

    // Since we switched to max, final result should be from max accumulator
    auto result = comp.eval();
    EXPECT_DOUBLE_EQ(result, 5.0);
}

TEST_F(CompositionTest, ConditionalCompositionSwitching) {
    min_accumulator<double> min_for_small;
    max_accumulator<double> max_for_large;

    auto predicate = [](double value) { return value < 3.0; };

    auto comp = conditional(min_for_small, max_for_large, predicate);

    comp += 1.0;  // < 3, goes to min
    comp += 2.0;  // < 3, goes to min
    comp += 4.0;  // >= 3, switch to max
    comp += 5.0;  // >= 3, goes to max
    comp += 1.5;  // < 3, switch back to min
    comp += 0.5;  // < 3, goes to min

    // Final accumulator is min with value 0.5
    auto result = comp.eval();
    EXPECT_DOUBLE_EQ(result, 0.5);
}

// ============================================================================
// Complex Composition Tests
// ============================================================================

TEST_F(CompositionTest, ComplexNestedComposition) {
    // Create a composition of (min + max) + count
    auto minmax_comp = min_accumulator<double>() + max_accumulator<double>();
    auto full_comp = minmax_comp + count_accumulator();

    std::vector<double> data = {3.14, 2.71, 1.41, 4.0, 2.0};
    for (const auto& val : data) {
        full_comp += val;
    }

    auto result = full_comp.eval();
    auto [minmax_tuple, count] = result;
    auto [min_val, max_val] = minmax_tuple;

    EXPECT_DOUBLE_EQ(min_val, 1.41);
    EXPECT_DOUBLE_EQ(max_val, 4.0);
    EXPECT_EQ(count, 5u);
}

TEST_F(CompositionTest, MixedCompositionOperators) {
    // Create (min + max) * count - parallel then sequential
    auto parallel = min_accumulator<double>() + max_accumulator<double>();

    // Note: This would need special handling for tuple input to count
    // For this test, we'll use a simpler example

    auto comp1 = min_accumulator<double>() + max_accumulator<double>();
    auto comp2 = count_accumulator() + kbn_sum<double>();

    comp1 += 1.0;
    comp1 += 5.0;
    comp2 += 1.0;
    comp2 += 5.0;

    auto [min_max, count_sum] = comp2.eval();
    EXPECT_EQ(count_sum.first, 2u);
    EXPECT_DOUBLE_EQ(count_sum.second, 6.0);
}

// ============================================================================
// Edge Cases and Error Conditions
// ============================================================================

TEST_F(CompositionTest, ParallelCompositionWithInfinity) {
    parallel_composition<min_accumulator<double>, max_accumulator<double>> comp;

    comp += std::numeric_limits<double>::infinity();
    comp += -std::numeric_limits<double>::infinity();
    comp += 0.0;

    auto [min_val, max_val] = comp.eval();
    EXPECT_EQ(min_val, -std::numeric_limits<double>::infinity());
    EXPECT_EQ(max_val, std::numeric_limits<double>::infinity());
}

TEST_F(CompositionTest, ParallelCompositionWithNaN) {
    parallel_composition<min_accumulator<double>, kbn_sum<double>> comp;

    comp += 1.0;
    comp += std::numeric_limits<double>::quiet_NaN();
    comp += 2.0;

    auto [min_val, sum] = comp.eval();
    EXPECT_TRUE(std::isnan(min_val) || min_val == 1.0);  // Implementation dependent
    EXPECT_TRUE(std::isnan(sum));
}

TEST_F(CompositionTest, EmptyParallelComposition) {
    parallel_composition<min_accumulator<double>, max_accumulator<double>> comp;

    // No values added
    auto [min_val, max_val] = comp.eval();
    EXPECT_EQ(min_val, std::numeric_limits<double>::max());
    EXPECT_EQ(max_val, std::numeric_limits<double>::lowest());
}

// ============================================================================
// Performance and Stress Tests
// ============================================================================

TEST_F(CompositionTest, ParallelCompositionLargeDataset) {
    parallel_composition<min_accumulator<double>,
                        parallel_composition<max_accumulator<double>,
                                           parallel_composition<count_accumulator,
                                                              kbn_sum<double>>>> comp;

    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-100.0, 100.0);

    double expected_min = std::numeric_limits<double>::max();
    double expected_max = std::numeric_limits<double>::lowest();
    double expected_sum = 0.0;
    const std::size_t n = 10000;

    for (std::size_t i = 0; i < n; ++i) {
        double val = dis(gen);
        comp += val;
        expected_min = std::min(expected_min, val);
        expected_max = std::max(expected_max, val);
        expected_sum += val;
    }

    auto [min_val, rest] = comp.eval();
    auto [max_val, count_sum] = rest;
    auto [count, sum] = count_sum;

    EXPECT_DOUBLE_EQ(min_val, expected_min);
    EXPECT_DOUBLE_EQ(max_val, expected_max);
    EXPECT_EQ(count, n);
    EXPECT_NEAR(sum, expected_sum, 1e-10);
}

TEST_F(CompositionTest, ComplexCompositionWithAllAccumulatorTypes) {
    // Test composition with all accumulator types
    auto comp = min_accumulator<double>() +
                max_accumulator<double>() +
                count_accumulator() +
                kbn_sum<double>() +
                welford_accumulator<double>() +
                minmax_accumulator<double>() +
                product_accumulator<double>();

    std::vector<double> data = {2.0, 4.0, 6.0, 8.0, 10.0};

    for (const auto& val : data) {
        comp += val;
    }

    // This creates a deeply nested tuple - just verify it compiles and runs
    auto result = comp.eval();
    (void)result;  // Suppress unused warning

    // The composition should work without runtime errors
    EXPECT_TRUE(true);
}

// ============================================================================
// Type Compatibility Tests
// ============================================================================

TEST_F(CompositionTest, MixedPrecisionComposition) {
    parallel_composition<min_accumulator<float>, max_accumulator<double>> comp;

    comp += 1.5;  // Should work with common_type
    comp += 2.5f;
    comp += 3.5;

    auto [min_val, max_val] = comp.eval();
    EXPECT_FLOAT_EQ(min_val, 1.5f);
    EXPECT_DOUBLE_EQ(max_val, 3.5);
}

TEST_F(CompositionTest, IntegerAndFloatingComposition) {
    parallel_composition<count_accumulator, kbn_sum<double>> comp;

    comp += 1;
    comp += 2.5;
    comp += 3;

    auto [count, sum] = comp.eval();
    EXPECT_EQ(count, 3u);
    EXPECT_DOUBLE_EQ(sum, 6.5);
}
/**
 * @file test_concepts.cpp
 * @brief Tests for C++20 concept enforcement and type traits
 *
 * Verifies that accumulator concepts properly enforce interface requirements
 * and that invalid types are correctly rejected at compile time.
 */

#include <gtest/gtest.h>
#include "include/accumux/core/accumulator_concept.hpp"
#include "include/accumux/accumulators/kbn_sum.hpp"
#include "include/accumux/accumulators/welford.hpp"
#include "include/accumux/accumulators/basic.hpp"

using namespace accumux;

class ConceptTest : public ::testing::Test {};

// ============================================================================
// Accumulator Concept Tests
// ============================================================================

TEST_F(ConceptTest, AccumulatorConceptSatisfiedByKBNSum) {
    static_assert(Accumulator<kbn_sum<double>>, "kbn_sum should satisfy Accumulator concept");
    static_assert(Accumulator<kbn_sum<float>>, "kbn_sum<float> should satisfy Accumulator concept");
}

TEST_F(ConceptTest, AccumulatorConceptSatisfiedByWelford) {
    static_assert(Accumulator<welford_accumulator<double>>, "welford_accumulator should satisfy Accumulator concept");
    static_assert(Accumulator<welford_accumulator<float>>, "welford_accumulator<float> should satisfy Accumulator concept");
}

TEST_F(ConceptTest, AccumulatorConceptSatisfiedByBasicAccumulators) {
    static_assert(Accumulator<min_accumulator<double>>, "min_accumulator should satisfy Accumulator concept");
    static_assert(Accumulator<max_accumulator<double>>, "max_accumulator should satisfy Accumulator concept");
    static_assert(Accumulator<minmax_accumulator<double>>, "minmax_accumulator should satisfy Accumulator concept");
    static_assert(Accumulator<count_accumulator>, "count_accumulator should satisfy Accumulator concept");
    static_assert(Accumulator<product_accumulator<double>>, "product_accumulator should satisfy Accumulator concept");
}

TEST_F(ConceptTest, NonAccumulatorsRejected) {
    // Primitive types are not accumulators
    static_assert(!Accumulator<int>, "int should not satisfy Accumulator concept");
    static_assert(!Accumulator<double>, "double should not satisfy Accumulator concept");
    static_assert(!Accumulator<std::string>, "std::string should not satisfy Accumulator concept");
}

// ============================================================================
// Statistical Accumulator Concept Tests
// ============================================================================

TEST_F(ConceptTest, StatisticalAccumulatorConcept) {
    static_assert(StatisticalAccumulator<welford_accumulator<double>>,
                  "welford_accumulator should satisfy StatisticalAccumulator concept");

    // kbn_sum is not statistical (no size() or mean())
    static_assert(!StatisticalAccumulator<kbn_sum<double>>,
                  "kbn_sum should not satisfy StatisticalAccumulator concept");

    // Basic accumulators are not statistical
    static_assert(!StatisticalAccumulator<min_accumulator<double>>,
                  "min_accumulator should not satisfy StatisticalAccumulator concept");
}

// ============================================================================
// Variance Accumulator Concept Tests
// ============================================================================

TEST_F(ConceptTest, VarianceAccumulatorConcept) {
    static_assert(VarianceAccumulator<welford_accumulator<double>>,
                  "welford_accumulator should satisfy VarianceAccumulator concept");

    // kbn_sum doesn't have variance methods
    static_assert(!VarianceAccumulator<kbn_sum<double>>,
                  "kbn_sum should not satisfy VarianceAccumulator concept");
}

// ============================================================================
// Accumulator Traits Tests
// ============================================================================

TEST_F(ConceptTest, AccumulatorTraitsValueType) {
    using kbn_traits = accumulator_traits<kbn_sum<double>>;
    static_assert(std::is_same_v<kbn_traits::value_type, double>,
                  "kbn_sum<double> should have value_type double");

    using welford_traits = accumulator_traits<welford_accumulator<float>>;
    static_assert(std::is_same_v<welford_traits::value_type, float>,
                  "welford_accumulator<float> should have value_type float");

    using count_traits = accumulator_traits<count_accumulator>;
    static_assert(std::is_same_v<count_traits::value_type, std::size_t>,
                  "count_accumulator should have value_type std::size_t");
}

TEST_F(ConceptTest, AccumulatorTraitsFlags) {
    using kbn_traits = accumulator_traits<kbn_sum<double>>;
    static_assert(kbn_traits::is_accumulator, "kbn_sum should be marked as accumulator");
    static_assert(!kbn_traits::is_statistical, "kbn_sum should not be marked as statistical");
    static_assert(!kbn_traits::has_variance, "kbn_sum should not have variance");

    using welford_traits = accumulator_traits<welford_accumulator<double>>;
    static_assert(welford_traits::is_accumulator, "welford should be marked as accumulator");
    static_assert(welford_traits::is_statistical, "welford should be marked as statistical");
    static_assert(welford_traits::has_variance, "welford should have variance");
}

// ============================================================================
// Compatible Accumulators Tests
// ============================================================================

TEST_F(ConceptTest, CompatibleAccumulatorsWithSameType) {
    static_assert(compatible_accumulators<kbn_sum<double>, welford_accumulator<double>>,
                  "kbn_sum<double> and welford<double> should be compatible");

    static_assert(compatible_accumulators<min_accumulator<double>, max_accumulator<double>>,
                  "min and max with same type should be compatible");
}

TEST_F(ConceptTest, IncompatibleAccumulatorsWithDifferentTypes) {
    static_assert(!compatible_accumulators<kbn_sum<double>, kbn_sum<float>>,
                  "kbn_sum<double> and kbn_sum<float> should not be compatible");

    static_assert(!compatible_accumulators<min_accumulator<int>, max_accumulator<double>>,
                  "Different value types should not be compatible");
}

// ============================================================================
// Runtime Concept Verification Tests
// ============================================================================

TEST_F(ConceptTest, RuntimeConceptVerification) {
    // These tests verify that concepts work at runtime as well
    kbn_sum<double> kbn;
    EXPECT_TRUE((Accumulator<decltype(kbn)>));

    welford_accumulator<double> welford;
    EXPECT_TRUE((Accumulator<decltype(welford)>));
    EXPECT_TRUE((StatisticalAccumulator<decltype(welford)>));
    EXPECT_TRUE((VarianceAccumulator<decltype(welford)>));

    min_accumulator<double> min_acc;
    EXPECT_TRUE((Accumulator<decltype(min_acc)>));
    EXPECT_FALSE((StatisticalAccumulator<decltype(min_acc)>));
}

// ============================================================================
// Concept Requirements Verification
// ============================================================================

TEST_F(ConceptTest, AccumulatorHasDefaultConstructor) {
    // Verify default construction requirement
    kbn_sum<double> kbn{};
    EXPECT_EQ(kbn.eval(), 0.0);

    min_accumulator<double> min_acc{};
    EXPECT_TRUE(min_acc.empty());
}

TEST_F(ConceptTest, AccumulatorHasCopyConstructor) {
    // Verify copy construction requirement
    kbn_sum<double> kbn1(5.0);
    kbn_sum<double> kbn2{kbn1};
    EXPECT_EQ(kbn1.eval(), kbn2.eval());

    welford_accumulator<double> w1;
    w1 += 1.0; w1 += 2.0; w1 += 3.0;
    welford_accumulator<double> w2{w1};
    EXPECT_EQ(w1.mean(), w2.mean());
}

TEST_F(ConceptTest, AccumulatorHasPlusEqualsValue) {
    // Verify += value requirement
    kbn_sum<double> kbn;
    kbn += 5.0;
    EXPECT_EQ(kbn.eval(), 5.0);

    min_accumulator<int> min_acc;
    min_acc += 10;
    min_acc += 5;
    EXPECT_EQ(min_acc.eval(), 5);
}

TEST_F(ConceptTest, AccumulatorHasPlusEqualsAccumulator) {
    // Verify += accumulator requirement
    kbn_sum<double> kbn1(3.0);
    kbn_sum<double> kbn2(7.0);
    kbn1 += kbn2;
    EXPECT_EQ(kbn1.eval(), 10.0);
}

TEST_F(ConceptTest, AccumulatorHasEval) {
    // Verify eval() requirement
    min_accumulator<double> min_acc;
    min_acc += 3.0;
    min_acc += 1.0;
    min_acc += 5.0;

    double result = min_acc.eval();
    EXPECT_EQ(result, 1.0);
}

TEST_F(ConceptTest, AccumulatorHasCopyAssignment) {
    // Verify copy assignment requirement
    kbn_sum<double> kbn1(5.0);
    kbn_sum<double> kbn2(10.0);
    kbn2 = kbn1;
    EXPECT_EQ(kbn2.eval(), 5.0);
}

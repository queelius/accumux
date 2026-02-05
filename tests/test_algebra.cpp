/**
 * @file test_algebra.cpp
 * @brief Tests for algebraic foundations (monoid laws, functors, etc.)
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "../include/accumux/core/algebra.hpp"
#include "../include/accumux/core/composition.hpp"
#include "../include/accumux/accumulators/kbn_sum.hpp"
#include "../include/accumux/accumulators/welford.hpp"
#include "../include/accumux/accumulators/basic.hpp"

using namespace accumux;
using namespace accumux::algebra;

class AlgebraTest : public ::testing::Test {
protected:
    static constexpr double EPSILON = 1e-10;
};

// ============================================================================
// Monoid Law Tests
// ============================================================================

TEST_F(AlgebraTest, MonoidConceptSatisfied) {
    static_assert(Monoid<kbn_sum<double>>);
    static_assert(Monoid<welford_accumulator<double>>);
    static_assert(Monoid<min_accumulator<double>>);
    static_assert(Monoid<max_accumulator<double>>);
    static_assert(Monoid<count_accumulator>);
}

TEST_F(AlgebraTest, KBNSumLeftIdentity) {
    EXPECT_TRUE(monoid_laws<kbn_sum<double>>::left_identity(42.0));
    EXPECT_TRUE(monoid_laws<kbn_sum<double>>::left_identity(-17.5));
    EXPECT_TRUE(monoid_laws<kbn_sum<double>>::left_identity(0.0));
}

TEST_F(AlgebraTest, KBNSumRightIdentity) {
    EXPECT_TRUE(monoid_laws<kbn_sum<double>>::right_identity(42.0));
    EXPECT_TRUE(monoid_laws<kbn_sum<double>>::right_identity(-17.5));
    EXPECT_TRUE(monoid_laws<kbn_sum<double>>::right_identity(0.0));
}

TEST_F(AlgebraTest, KBNSumAssociativity) {
    EXPECT_TRUE(monoid_laws<kbn_sum<double>>::associativity(1.0, 2.0, 3.0));
    EXPECT_TRUE(monoid_laws<kbn_sum<double>>::associativity(100.0, 0.001, 0.000001));
}

TEST_F(AlgebraTest, CountAccumulatorMonoidLaws) {
    std::vector<int> test_values = {1, 2, 3, 4, 5};
    EXPECT_TRUE(algebraic_properties<count_accumulator>::verify_monoid(test_values));
}

TEST_F(AlgebraTest, MinAccumulatorMonoidLaws) {
    std::vector<double> test_values = {5.0, 2.0, 8.0, 1.0, 9.0};
    EXPECT_TRUE(algebraic_properties<min_accumulator<double>>::verify_monoid(test_values));
}

TEST_F(AlgebraTest, EvalHomomorphism) {
    EXPECT_TRUE(algebraic_properties<kbn_sum<double>>::verify_eval_homomorphism(3.0, 7.0));
}

// ============================================================================
// Functor Tests
// ============================================================================

TEST_F(AlgebraTest, FmapBasic) {
    kbn_sum<double> sum;
    sum += 10.0;

    auto doubled = fmap([](double x) { return x * 2; }, sum);
    EXPECT_NEAR(doubled.eval(), 20.0, EPSILON);
}

TEST_F(AlgebraTest, FmapAccumulation) {
    auto squared_sum = fmap([](double x) { return x * x; }, kbn_sum<double>());

    squared_sum += 3.0;  // Accumulates 3, returns 9
    EXPECT_NEAR(squared_sum.eval(), 9.0, EPSILON);

    squared_sum += 4.0;  // Accumulates 4, sum is 7, returns 49
    EXPECT_NEAR(squared_sum.eval(), 49.0, EPSILON);
}

TEST_F(AlgebraTest, FmapPreservesAccumulatorInterface) {
    auto mapped = fmap([](double x) { return x + 1; }, kbn_sum<double>());
    static_assert(Accumulator<decltype(mapped)>);
}

TEST_F(AlgebraTest, FmapChaining) {
    auto transformed = fmap([](double x) { return x * 2; },
                       fmap([](double x) { return x + 1; }, kbn_sum<double>()));

    transformed += 5.0;  // 5 -> sum=5 -> +1=6 -> *2=12
    EXPECT_NEAR(transformed.eval(), 12.0, EPSILON);
}

// ============================================================================
// Pure and Applicative Tests
// ============================================================================

TEST_F(AlgebraTest, PureBasic) {
    auto constant = pure(42.0);
    static_assert(Accumulator<decltype(constant)>);

    constant += 100.0;  // Ignored
    constant += 200.0;  // Ignored

    EXPECT_NEAR(constant.eval(), 42.0, EPSILON);
}

TEST_F(AlgebraTest, PureAsIdentity) {
    auto identity = pure(0.0);

    identity += 1.0;
    identity += 2.0;

    // Pure ignores all input
    EXPECT_NEAR(identity.eval(), 0.0, EPSILON);
}

// ============================================================================
// Fold Tests
// ============================================================================

TEST_F(AlgebraTest, FoldBasic) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};

    auto result = fold<kbn_sum<double>>(data.begin(), data.end());
    EXPECT_NEAR(result.eval(), 15.0, EPSILON);
}

TEST_F(AlgebraTest, FoldWithInitial) {
    std::vector<double> data = {1.0, 2.0, 3.0};

    kbn_sum<double> init;
    init += 10.0;

    auto result = fold(init, data.begin(), data.end());
    EXPECT_NEAR(result.eval(), 16.0, EPSILON);
}

TEST_F(AlgebraTest, FoldWelford) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};

    auto result = fold<welford_accumulator<double>>(data.begin(), data.end());
    EXPECT_EQ(result.size(), 5);
    EXPECT_NEAR(result.mean(), 3.0, EPSILON);
}

TEST_F(AlgebraTest, FoldMinMax) {
    std::vector<double> data = {5.0, 2.0, 8.0, 1.0, 9.0, 3.0};

    auto min_result = fold<min_accumulator<double>>(data.begin(), data.end());
    auto max_result = fold<max_accumulator<double>>(data.begin(), data.end());

    EXPECT_NEAR(min_result.eval(), 1.0, EPSILON);
    EXPECT_NEAR(max_result.eval(), 9.0, EPSILON);
}

// ============================================================================
// Homomorphism Tests
// ============================================================================

TEST_F(AlgebraTest, IdentityHomomorphism) {
    identity_homomorphism<kbn_sum<double>> id;

    kbn_sum<double> acc;
    acc += 5.0;

    auto result = id(acc);
    EXPECT_NEAR(result.eval(), 5.0, EPSILON);
}

TEST_F(AlgebraTest, EvalHomomorphismType) {
    eval_homomorphism<kbn_sum<double>> eval_h;

    kbn_sum<double> acc;
    acc += 7.0;

    double result = eval_h(acc);
    EXPECT_NEAR(result, 7.0, EPSILON);
}

TEST_F(AlgebraTest, ComposedHomomorphism) {
    auto h1 = [](const kbn_sum<double>& acc) {
        kbn_sum<double> result;
        result += acc.eval() * 2;
        return result;
    };

    auto h2 = [](const kbn_sum<double>& acc) {
        kbn_sum<double> result;
        result += acc.eval() + 1;
        return result;
    };

    auto composed = compose(h1, h2);

    kbn_sum<double> acc;
    acc += 5.0;

    auto result = composed(acc);
    // h2(acc) = 5 + 1 = 6, h1(6) = 6 * 2 = 12
    EXPECT_NEAR(result.eval(), 12.0, EPSILON);
}

// ============================================================================
// Algebraic Traits Tests
// ============================================================================

TEST_F(AlgebraTest, AlgebraicTraitsMonoid) {
    using traits = algebraic_traits<kbn_sum<double>>;

    static_assert(traits::is_monoid);
    static_assert(traits::is_semigroup);
    static_assert(traits::has_identity);
    static_assert(traits::structure == algebraic_structure::monoid);
}

TEST_F(AlgebraTest, ParallelCompositionIsMonoid) {
    using comp_type = parallel_composition<kbn_sum<double>, count_accumulator>;
    static_assert(Monoid<comp_type>);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(AlgebraTest, AlgebraicPipeline) {
    // Build a pipeline: accumulate -> double -> add 10
    auto pipeline = fmap([](double x) { return x + 10; },
                    fmap([](double x) { return x * 2; },
                         kbn_sum<double>()));

    pipeline += 5.0;   // sum=5 -> *2=10 -> +10=20
    pipeline += 10.0;  // sum=15 -> *2=30 -> +10=40

    EXPECT_NEAR(pipeline.eval(), 40.0, EPSILON);
}

TEST_F(AlgebraTest, CompositionWithFmap) {
    // Compose two accumulators, then map the result
    auto comp = kbn_sum<double>() + count_accumulator();

    comp += 1.0;
    comp += 2.0;
    comp += 3.0;

    auto [sum, count] = comp.eval();
    EXPECT_NEAR(sum, 6.0, EPSILON);
    EXPECT_EQ(count, 3);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

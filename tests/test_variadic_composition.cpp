/**
 * @file test_variadic_composition.cpp
 * @brief Tests for variadic parallel composition
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "../include/accumux/core/variadic_composition.hpp"
#include "../include/accumux/accumulators/kbn_sum.hpp"
#include "../include/accumux/accumulators/welford.hpp"
#include "../include/accumux/accumulators/basic.hpp"

using namespace accumux;

class VariadicCompositionTest : public ::testing::Test {
protected:
    static constexpr double EPSILON = 1e-10;
};

TEST_F(VariadicCompositionTest, BasicConstruction) {
    auto comp = make_parallel(
        kbn_sum<double>(),
        welford_accumulator<double>(),
        min_accumulator<double>()
    );

    static_assert(comp.accumulator_count == 3);
}

TEST_F(VariadicCompositionTest, ValueAccumulation) {
    auto comp = make_parallel(
        kbn_sum<double>(),
        count_accumulator(),
        min_accumulator<double>(),
        max_accumulator<double>()
    );

    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    for (double v : data) {
        comp += v;
    }

    EXPECT_NEAR(comp.get<0>().eval(), 15.0, EPSILON);
    EXPECT_EQ(comp.get<1>().eval(), 5);
    EXPECT_NEAR(comp.get<2>().eval(), 1.0, EPSILON);
    EXPECT_NEAR(comp.get<3>().eval(), 5.0, EPSILON);
}

TEST_F(VariadicCompositionTest, EvalReturnsTuple) {
    auto comp = make_parallel(
        kbn_sum<double>(),
        min_accumulator<double>()
    );

    comp += 10.0;
    comp += 20.0;

    auto [sum, minimum] = comp.eval();
    EXPECT_NEAR(sum, 30.0, EPSILON);
    EXPECT_NEAR(minimum, 10.0, EPSILON);
}

TEST_F(VariadicCompositionTest, CombineCompositions) {
    auto comp1 = make_parallel(kbn_sum<double>(), count_accumulator());
    auto comp2 = make_parallel(kbn_sum<double>(), count_accumulator());

    comp1 += 1.0;
    comp1 += 2.0;

    comp2 += 3.0;
    comp2 += 4.0;

    comp1 += comp2;

    EXPECT_NEAR(comp1.get<0>().eval(), 10.0, EPSILON);
    EXPECT_EQ(comp1.get<1>().eval(), 4);
}

TEST_F(VariadicCompositionTest, ForEach) {
    auto comp = make_parallel(
        kbn_sum<double>(),
        kbn_sum<double>(),
        kbn_sum<double>()
    );

    comp += 5.0;

    int count = 0;
    comp.for_each([&count](const auto& acc) {
        EXPECT_NEAR(acc.eval(), 5.0, 1e-10);
        ++count;
    });

    EXPECT_EQ(count, 3);
}

TEST_F(VariadicCompositionTest, Transform) {
    auto comp = make_parallel(
        kbn_sum<double>(),
        count_accumulator()
    );

    comp += 10.0;
    comp += 20.0;
    comp += 30.0;

    auto results = comp.transform([](const auto& acc) {
        return static_cast<double>(acc.eval());
    });

    EXPECT_NEAR(std::get<0>(results), 60.0, EPSILON);
    EXPECT_NEAR(std::get<1>(results), 3.0, EPSILON);
}

TEST_F(VariadicCompositionTest, ManyAccumulators) {
    auto comp = make_parallel(
        kbn_sum<double>(),
        welford_accumulator<double>(),
        min_accumulator<double>(),
        max_accumulator<double>(),
        count_accumulator(),
        minmax_accumulator<double>()
    );

    static_assert(comp.accumulator_count == 6);

    for (int i = 1; i <= 100; ++i) {
        comp += static_cast<double>(i);
    }

    EXPECT_NEAR(comp.get<0>().eval(), 5050.0, EPSILON);
    EXPECT_EQ(comp.get<4>().eval(), 100);
    EXPECT_NEAR(comp.get<2>().eval(), 1.0, EPSILON);
    EXPECT_NEAR(comp.get<3>().eval(), 100.0, EPSILON);
}

TEST_F(VariadicCompositionTest, EmptyComposition) {
    auto comp = make_parallel(kbn_sum<double>());

    static_assert(comp.accumulator_count == 1);

    comp += 42.0;
    EXPECT_NEAR(comp.get<0>().eval(), 42.0, EPSILON);
}

TEST_F(VariadicCompositionTest, GetByType) {
    auto comp = make_parallel(
        kbn_sum<double>(),
        count_accumulator()
    );

    comp += 5.0;
    comp += 10.0;

    const auto& sum = comp.get<kbn_sum<double>>();
    const auto& cnt = comp.get<count_accumulator>();

    EXPECT_NEAR(sum.eval(), 15.0, EPSILON);
    EXPECT_EQ(cnt.eval(), 2);
}

// Test that variadic composition satisfies Accumulator concept
TEST_F(VariadicCompositionTest, ConceptCompliance) {
    using comp_type = variadic_parallel_composition<kbn_sum<double>, count_accumulator>;
    static_assert(Accumulator<comp_type>);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

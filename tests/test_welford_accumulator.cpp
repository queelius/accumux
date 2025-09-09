#include <gtest/gtest.h>
#include <vector>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include "welford_accumulator.hpp"
#include "kbn_sum.hpp"

using namespace algebraic_accumulators;
using namespace algebraic_accumulator;

class WelfordAccumulatorTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
    
    // Helper function to check floating point equality with tolerance
    template<typename T>
    bool nearly_equal(T a, T b, T tolerance = 1e-12) {
        return std::abs(a - b) <= tolerance;
    }
    
    // Helper to calculate expected variance
    template<typename Container>
    double calculate_expected_variance(const Container& values, bool sample = false) {
        if (values.empty()) return 0.0;
        
        double mean = 0.0;
        for (auto val : values) {
            mean += val;
        }
        mean /= values.size();
        
        double variance = 0.0;
        for (auto val : values) {
            variance += (val - mean) * (val - mean);
        }
        
        if (sample && values.size() > 1) {
            variance /= (values.size() - 1);
        } else {
            variance /= values.size();
        }
        
        return variance;
    }
    
    template<typename Container>
    double calculate_expected_mean(const Container& values) {
        if (values.empty()) return 0.0;
        
        double sum = 0.0;
        for (auto val : values) {
            sum += val;
        }
        return sum / values.size();
    }
};

// Test default constructor
TEST_F(WelfordAccumulatorTest, DefaultConstructor) {
    welford_accumulator<kbn_sum<double>> acc;
    
    EXPECT_EQ(acc.count, 0u);
    EXPECT_EQ(acc.mean(), 0.0);
    EXPECT_EQ(acc.size(), 0u);
    EXPECT_EQ(acc.sum(), 0.0);
    EXPECT_EQ(static_cast<double>(acc), 0.0);
}

// Test constructor with initial value
TEST_F(WelfordAccumulatorTest, ValueConstructor) {
    welford_accumulator<kbn_sum<double>> acc(5.0);
    
    EXPECT_EQ(acc.count, 1u);
    EXPECT_EQ(acc.mean(), 5.0);
    EXPECT_EQ(acc.size(), 1u);
    EXPECT_EQ(acc.sum(), 5.0);
    EXPECT_EQ(static_cast<double>(acc), 5.0);
    EXPECT_EQ(acc.variance(), 0.0); // Single value has no variance
}

// Test copy constructor
TEST_F(WelfordAccumulatorTest, CopyConstructor) {
    welford_accumulator<kbn_sum<double>> acc1(3.14);
    welford_accumulator<kbn_sum<double>> acc2(acc1);
    
    EXPECT_EQ(acc2.count, acc1.count);
    EXPECT_EQ(acc2.mean(), acc1.mean());
    EXPECT_EQ(acc2.variance(), acc1.variance());
}

// Test adding single values
TEST_F(WelfordAccumulatorTest, AddingSingleValues) {
    welford_accumulator<kbn_sum<double>> acc;
    
    acc += 1.0;
    EXPECT_EQ(acc.count, 1u);
    EXPECT_EQ(acc.mean(), 1.0);
    EXPECT_EQ(acc.sum(), 1.0);
    
    acc += 3.0;
    EXPECT_EQ(acc.count, 2u);
    EXPECT_EQ(acc.mean(), 2.0);
    EXPECT_EQ(acc.sum(), 4.0);
    
    acc += 5.0;
    EXPECT_EQ(acc.count, 3u);
    EXPECT_EQ(acc.mean(), 3.0);
    EXPECT_EQ(acc.sum(), 9.0);
}

// Test variance calculation
TEST_F(WelfordAccumulatorTest, VarianceCalculation) {
    welford_accumulator<kbn_sum<double>> acc;
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    for (double val : values) {
        acc += val;
    }
    
    double expected_mean = calculate_expected_mean(values);
    double expected_variance = calculate_expected_variance(values, false);
    double expected_sample_variance = calculate_expected_variance(values, true);
    
    EXPECT_TRUE(nearly_equal(acc.mean(), expected_mean, 1e-10));
    EXPECT_TRUE(nearly_equal(acc.variance(), expected_variance, 1e-10));
    EXPECT_TRUE(nearly_equal(acc.sample_variance(), expected_sample_variance, 1e-10));
}

// Test with floating point values
TEST_F(WelfordAccumulatorTest, FloatingPointValues) {
    welford_accumulator<kbn_sum<double>> acc;
    std::vector<double> values = {1.1, 2.2, 3.3, 4.4, 5.5};
    
    for (double val : values) {
        acc += val;
    }
    
    double expected_mean = calculate_expected_mean(values);
    double expected_variance = calculate_expected_variance(values, false);
    
    EXPECT_TRUE(nearly_equal(acc.mean(), expected_mean, 1e-10));
    EXPECT_TRUE(nearly_equal(acc.variance(), expected_variance, 1e-10));
    EXPECT_EQ(acc.size(), 5u);
    EXPECT_TRUE(nearly_equal(acc.sum(), 16.5, 1e-10));
}

// Test with negative values
TEST_F(WelfordAccumulatorTest, NegativeValues) {
    welford_accumulator<kbn_sum<double>> acc;
    std::vector<double> values = {-1.0, -2.0, -3.0, -4.0, -5.0};
    
    for (double val : values) {
        acc += val;
    }
    
    double expected_mean = calculate_expected_mean(values);
    double expected_variance = calculate_expected_variance(values, false);
    
    EXPECT_TRUE(nearly_equal(acc.mean(), expected_mean, 1e-10));
    EXPECT_TRUE(nearly_equal(acc.variance(), expected_variance, 1e-10));
    EXPECT_EQ(acc.sum(), -15.0);
}

// Test with mixed positive and negative values
TEST_F(WelfordAccumulatorTest, MixedSignValues) {
    welford_accumulator<kbn_sum<double>> acc;
    std::vector<double> values = {-2.0, 1.0, -1.0, 2.0, 0.0};
    
    for (double val : values) {
        acc += val;
    }
    
    double expected_mean = calculate_expected_mean(values);
    double expected_variance = calculate_expected_variance(values, false);
    
    EXPECT_TRUE(nearly_equal(acc.mean(), expected_mean, 1e-10));
    EXPECT_TRUE(nearly_equal(acc.variance(), expected_variance, 1e-10));
    EXPECT_TRUE(nearly_equal(acc.sum(), 0.0, 1e-10));
}

// Test sample variance vs population variance
TEST_F(WelfordAccumulatorTest, SampleVsPopulationVariance) {
    welford_accumulator<kbn_sum<double>> acc;
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0};
    
    for (double val : values) {
        acc += val;
    }
    
    // Population variance: divide by n
    double population_var = acc.variance();
    // Sample variance: divide by n-1
    double sample_var = acc.sample_variance();
    
    EXPECT_GT(sample_var, population_var);
    EXPECT_TRUE(nearly_equal(sample_var, population_var * 4.0 / 3.0, 1e-10));
}

// Test single value edge case for sample variance
TEST_F(WelfordAccumulatorTest, SingleValueSampleVariance) {
    welford_accumulator<kbn_sum<double>> acc;
    acc += 5.0;
    
    EXPECT_EQ(acc.variance(), 0.0);
    
    // Sample variance with n=1 should divide by 0, which gives infinity or undefined behavior
    // But mathematically it's undefined, so we expect infinity
    double sample_var = acc.sample_variance();
    EXPECT_TRUE(std::isinf(sample_var) || std::isnan(sample_var));
}

// Test empty accumulator edge cases
TEST_F(WelfordAccumulatorTest, EmptyAccumulatorEdgeCases) {
    welford_accumulator<kbn_sum<double>> acc;
    
    // For empty accumulator with count=0:
    // - variance calculation: M2/count = 0/0 = NaN
    // - sample variance: M2/(count-1) = 0/(-1) = 0
    EXPECT_TRUE(std::isnan(acc.variance()));
    EXPECT_EQ(acc.sample_variance(), 0.0);
    EXPECT_EQ(acc.mean(), 0.0);
    EXPECT_EQ(acc.sum(), 0.0);
    EXPECT_EQ(acc.size(), 0u);
}

// Test large numbers for numerical stability
TEST_F(WelfordAccumulatorTest, LargeNumbers) {
    welford_accumulator<kbn_sum<double>> acc;
    std::vector<double> values = {1e10, 1e10 + 1, 1e10 + 2, 1e10 + 3};
    
    for (double val : values) {
        acc += val;
    }
    
    EXPECT_TRUE(nearly_equal(acc.mean(), 1e10 + 1.5, 1e-6));
    EXPECT_GT(acc.variance(), 0.0);
}

// Test many small values for precision
TEST_F(WelfordAccumulatorTest, ManySmallValues) {
    welford_accumulator<kbn_sum<double>> acc;
    std::vector<double> values(1000, 0.001);
    
    for (double val : values) {
        acc += val;
    }
    
    EXPECT_TRUE(nearly_equal(acc.mean(), 0.001, 1e-10));
    EXPECT_TRUE(nearly_equal(acc.sum(), 1.0, 1e-10));
    EXPECT_TRUE(nearly_equal(acc.variance(), 0.0, 1e-15)); // All values are identical
    EXPECT_EQ(acc.size(), 1000u);
}

// Test using kbn_welford_accumulate alias
TEST_F(WelfordAccumulatorTest, KBNWelfordAlias) {
    kbn_welford_accumulate<double> acc;
    
    acc += 1.0;
    acc += 2.0;
    acc += 3.0;
    
    EXPECT_EQ(acc.size(), 3u);
    EXPECT_EQ(acc.mean(), 2.0);
    EXPECT_EQ(acc.sum(), 6.0);
}

// Test standalone function interfaces
TEST_F(WelfordAccumulatorTest, StandaloneFunctions) {
    welford_accumulator<kbn_sum<double>> acc;
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    for (double val : values) {
        acc += val;
    }
    
    EXPECT_EQ(mean(acc), acc.mean());
    EXPECT_EQ(variance(acc), acc.variance());
    EXPECT_EQ(sample_variance(acc), acc.sample_variance());
    EXPECT_EQ(size(acc), acc.size());
    EXPECT_EQ(sum(acc), acc.sum());
}

// Test with different underlying accumulator types
TEST_F(WelfordAccumulatorTest, DifferentAccumulatorTypes) {
    // Test with kbn_sum as accumulator (which is the intended use case)
    welford_accumulator<kbn_sum<double>> kbn_acc;
    
    kbn_acc += 1.0;
    kbn_acc += 2.0;
    kbn_acc += 3.0;
    
    EXPECT_EQ(kbn_acc.mean(), 2.0);
    EXPECT_EQ(kbn_acc.sum(), 6.0);
    EXPECT_EQ(kbn_acc.size(), 3u);
}

// Test numerical precision compared to naive calculation
TEST_F(WelfordAccumulatorTest, NumericalPrecision) {
    welford_accumulator<kbn_sum<double>> acc;
    
    // Values designed to test numerical precision
    std::vector<double> values;
    for (int i = 0; i < 1000; ++i) {
        values.push_back(1.0 + i * 1e-10);
    }
    
    for (double val : values) {
        acc += val;
    }
    
    double expected_mean = calculate_expected_mean(values);
    double expected_variance = calculate_expected_variance(values, false);
    
    EXPECT_TRUE(nearly_equal(acc.mean(), expected_mean, 1e-8));
    EXPECT_TRUE(nearly_equal(acc.variance(), expected_variance, 1e-15));
}

// Test random values stress test
TEST_F(WelfordAccumulatorTest, RandomValuesStressTest) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-100.0, 100.0);
    
    std::vector<double> random_values(10000);
    for (auto& val : random_values) {
        val = dis(gen);
    }
    
    welford_accumulator<kbn_sum<double>> acc;
    for (double val : random_values) {
        acc += val;
    }
    
    double expected_mean = calculate_expected_mean(random_values);
    double expected_variance = calculate_expected_variance(random_values, false);
    double expected_sample_variance = calculate_expected_variance(random_values, true);
    
    EXPECT_TRUE(nearly_equal(acc.mean(), expected_mean, 1e-10));
    EXPECT_TRUE(nearly_equal(acc.variance(), expected_variance, 1e-8));
    EXPECT_TRUE(nearly_equal(acc.sample_variance(), expected_sample_variance, 1e-8));
    EXPECT_EQ(acc.size(), 10000u);
}

// Test incremental calculation property
TEST_F(WelfordAccumulatorTest, IncrementalCalculation) {
    welford_accumulator<kbn_sum<double>> acc;
    std::vector<double> values = {1.0, 4.0, 7.0, 2.0, 8.0};
    
    std::vector<double> means, variances;
    
    for (double val : values) {
        acc += val;
        means.push_back(acc.mean());
        variances.push_back(acc.variance());
    }
    
    // Check that incremental calculations are consistent
    EXPECT_EQ(means[0], 1.0);
    EXPECT_EQ(means[1], 2.5);
    EXPECT_TRUE(nearly_equal(means[2], 4.0, 1e-10));
    EXPECT_TRUE(nearly_equal(means[3], 3.5, 1e-10));
    EXPECT_TRUE(nearly_equal(means[4], 4.4, 1e-10));
    
    // Variance should be zero for single element
    EXPECT_EQ(variances[0], 0.0);
    EXPECT_GT(variances[4], 0.0);
}

// Test move semantics in delta2 calculation
TEST_F(WelfordAccumulatorTest, MoveSemanticsInDelta2) {
    welford_accumulator<kbn_sum<double>> acc;
    
    // Add values to test the move operation in delta2 calculation
    acc += 10.0;
    acc += 20.0;
    
    // The move operation should work correctly without affecting results
    EXPECT_EQ(acc.mean(), 15.0);
    EXPECT_EQ(acc.variance(), 25.0); // variance of {10, 20} is 25
}

// Test edge case with identical values
TEST_F(WelfordAccumulatorTest, IdenticalValues) {
    welford_accumulator<kbn_sum<double>> acc;
    
    for (int i = 0; i < 100; ++i) {
        acc += 42.0;
    }
    
    EXPECT_EQ(acc.mean(), 42.0);
    EXPECT_EQ(acc.sum(), 4200.0);
    EXPECT_EQ(acc.variance(), 0.0); // All identical values should have zero variance
    EXPECT_EQ(acc.sample_variance(), 0.0);
    EXPECT_EQ(acc.size(), 100u);
}
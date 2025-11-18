#include <gtest/gtest.h>
#include <vector>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include "include/accumux/accumulators/welford.hpp"
#include "include/accumux/accumulators/kbn_sum.hpp"

using namespace accumux;

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
    welford_accumulator<double> acc;
    
    EXPECT_EQ(acc.size(), 0u);
    EXPECT_EQ(acc.mean(), 0.0);
    EXPECT_EQ(acc.size(), 0u);
    EXPECT_EQ(acc.sum(), 0.0);
    EXPECT_EQ(static_cast<double>(acc), 0.0);
}

// Test constructor with initial value
TEST_F(WelfordAccumulatorTest, ValueConstructor) {
    welford_accumulator<double> acc(5.0);
    
    EXPECT_EQ(acc.size(), 1u);
    EXPECT_EQ(acc.mean(), 5.0);
    EXPECT_EQ(acc.size(), 1u);
    EXPECT_EQ(acc.sum(), 5.0);
    EXPECT_EQ(static_cast<double>(acc), 5.0);
    EXPECT_EQ(acc.variance(), 0.0); // Single value has no variance
}

// Test copy constructor
TEST_F(WelfordAccumulatorTest, CopyConstructor) {
    welford_accumulator<double> acc1(3.14);
    welford_accumulator<double> acc2(acc1);
    
    EXPECT_EQ(acc2.size(), acc1.size());
    EXPECT_EQ(acc2.mean(), acc1.mean());
    EXPECT_EQ(acc2.variance(), acc1.variance());
}

// Test adding single values
TEST_F(WelfordAccumulatorTest, AddingSingleValues) {
    welford_accumulator<double> acc;
    
    acc += 1.0;
    EXPECT_EQ(acc.size(), 1u);
    EXPECT_EQ(acc.mean(), 1.0);
    EXPECT_EQ(acc.sum(), 1.0);
    
    acc += 3.0;
    EXPECT_EQ(acc.size(), 2u);
    EXPECT_EQ(acc.mean(), 2.0);
    EXPECT_EQ(acc.sum(), 4.0);
    
    acc += 5.0;
    EXPECT_EQ(acc.size(), 3u);
    EXPECT_EQ(acc.mean(), 3.0);
    EXPECT_EQ(acc.sum(), 9.0);
}

// Test variance calculation
TEST_F(WelfordAccumulatorTest, VarianceCalculation) {
    welford_accumulator<double> acc;
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
    welford_accumulator<double> acc;
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
    welford_accumulator<double> acc;
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
    welford_accumulator<double> acc;
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
    welford_accumulator<double> acc;
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
    welford_accumulator<double> acc;
    acc += 5.0;
    
    EXPECT_EQ(acc.variance(), 0.0);
    
    // Sample variance with n=1 should return 0 in our implementation
    double sample_var = acc.sample_variance();
    EXPECT_EQ(sample_var, 0.0);
}

// Test empty accumulator edge cases
TEST_F(WelfordAccumulatorTest, EmptyAccumulatorEdgeCases) {
    welford_accumulator<double> acc;
    
    // For empty accumulator, all values should be 0
    EXPECT_EQ(acc.variance(), 0.0);
    EXPECT_EQ(acc.sample_variance(), 0.0);
    EXPECT_EQ(acc.mean(), 0.0);
    EXPECT_EQ(acc.sum(), 0.0);
    EXPECT_EQ(acc.size(), 0u);
}

// Test large numbers for numerical stability
TEST_F(WelfordAccumulatorTest, LargeNumbers) {
    welford_accumulator<double> acc;
    std::vector<double> values = {1e10, 1e10 + 1, 1e10 + 2, 1e10 + 3};
    
    for (double val : values) {
        acc += val;
    }
    
    EXPECT_TRUE(nearly_equal(acc.mean(), 1e10 + 1.5, 1e-6));
    EXPECT_GT(acc.variance(), 0.0);
}

// Test many small values for precision
TEST_F(WelfordAccumulatorTest, ManySmallValues) {
    welford_accumulator<double> acc;
    std::vector<double> values(1000, 0.001);
    
    for (double val : values) {
        acc += val;
    }
    
    EXPECT_TRUE(nearly_equal(acc.mean(), 0.001, 1e-10));
    EXPECT_TRUE(nearly_equal(acc.sum(), 1.0, 1e-10));
    EXPECT_TRUE(nearly_equal(acc.variance(), 0.0, 1e-15)); // All values are identical
    EXPECT_EQ(acc.size(), 1000u);
}

// Test using make_welford_accumulator factory function
TEST_F(WelfordAccumulatorTest, FactoryFunction) {
    auto acc = make_welford_accumulator<double>();
    
    acc += 1.0;
    acc += 2.0;
    acc += 3.0;
    
    EXPECT_EQ(acc.size(), 3u);
    EXPECT_EQ(acc.mean(), 2.0);
    EXPECT_EQ(acc.sum(), 6.0);
}

// Test iterator-based convenience functions
TEST_F(WelfordAccumulatorTest, IteratorBasedFunctions) {
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    // Test the convenience functions for computing statistics from iterators
    double computed_mean = mean(values.begin(), values.end());
    double computed_variance = variance(values.begin(), values.end());
    
    welford_accumulator<double> acc;
    for (double val : values) {
        acc += val;
    }
    
    EXPECT_EQ(computed_mean, acc.mean());
    EXPECT_EQ(computed_variance, acc.variance());
}

// Test standard deviation functions
TEST_F(WelfordAccumulatorTest, StandardDeviation) {
    welford_accumulator<double> acc;
    
    acc += 1.0;
    acc += 2.0;
    acc += 3.0;
    
    EXPECT_EQ(acc.mean(), 2.0);
    EXPECT_EQ(acc.sum(), 6.0);
    EXPECT_EQ(acc.size(), 3u);
    
    // Test that std_dev is square root of variance
    EXPECT_DOUBLE_EQ(acc.std_dev(), std::sqrt(acc.variance()));
    EXPECT_DOUBLE_EQ(acc.sample_std_dev(), std::sqrt(acc.sample_variance()));
}

// Test numerical precision compared to naive calculation
TEST_F(WelfordAccumulatorTest, NumericalPrecision) {
    welford_accumulator<double> acc;
    
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
    
    welford_accumulator<double> acc;
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
    welford_accumulator<double> acc;
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
    welford_accumulator<double> acc;
    
    // Add values to test the move operation in delta2 calculation
    acc += 10.0;
    acc += 20.0;
    
    // The move operation should work correctly without affecting results
    EXPECT_EQ(acc.mean(), 15.0);
    EXPECT_EQ(acc.variance(), 25.0); // variance of {10, 20} is 25
}

// Test edge case with identical values
TEST_F(WelfordAccumulatorTest, IdenticalValues) {
    welford_accumulator<double> acc;
    
    for (int i = 0; i < 100; ++i) {
        acc += 42.0;
    }
    
    EXPECT_EQ(acc.mean(), 42.0);
    EXPECT_EQ(acc.sum(), 4200.0);
    EXPECT_EQ(acc.variance(), 0.0); // All identical values should have zero variance
    EXPECT_EQ(acc.sample_variance(), 0.0);
    EXPECT_EQ(acc.size(), 100u);
}

// Test empty() method - basic behavior
TEST_F(WelfordAccumulatorTest, EmptyMethod) {
    welford_accumulator<double> acc;

    // Should be empty initially
    EXPECT_TRUE(acc.empty());
    EXPECT_EQ(acc.size(), 0u);

    // Should not be empty after adding a value
    acc += 5.0;
    EXPECT_FALSE(acc.empty());
    EXPECT_EQ(acc.size(), 1u);

    // Should still not be empty with more values
    acc += 10.0;
    acc += 15.0;
    EXPECT_FALSE(acc.empty());
    EXPECT_EQ(acc.size(), 3u);
}

// Test combining empty accumulators
TEST_F(WelfordAccumulatorTest, CombineWithBothEmpty) {
    welford_accumulator<double> acc1, acc2;

    EXPECT_TRUE(acc1.empty());
    EXPECT_TRUE(acc2.empty());

    acc1 += acc2;

    EXPECT_TRUE(acc1.empty());
    EXPECT_EQ(acc1.size(), 0u);
    EXPECT_EQ(acc1.mean(), 0.0);
}

// Test combining empty with non-empty accumulator
TEST_F(WelfordAccumulatorTest, CombineEmptyWithNonEmpty) {
    welford_accumulator<double> acc1, acc2;

    // acc2 has data
    acc2 += 5.0;
    acc2 += 10.0;
    acc2 += 15.0;

    EXPECT_TRUE(acc1.empty());
    EXPECT_FALSE(acc2.empty());

    // Combine empty into non-empty
    acc1 += acc2;

    EXPECT_FALSE(acc1.empty());
    EXPECT_EQ(acc1.size(), 3u);
    EXPECT_EQ(acc1.mean(), acc2.mean());
}

// Test combining non-empty with empty accumulator
TEST_F(WelfordAccumulatorTest, CombineNonEmptyWithEmpty) {
    welford_accumulator<double> acc1, acc2;

    // acc1 has data
    acc1 += 5.0;
    acc1 += 10.0;

    EXPECT_FALSE(acc1.empty());
    EXPECT_TRUE(acc2.empty());

    double original_mean = acc1.mean();
    std::size_t original_size = acc1.size();

    // Combine non-empty with empty
    acc1 += acc2;

    EXPECT_FALSE(acc1.empty());
    EXPECT_EQ(acc1.size(), original_size);
    EXPECT_EQ(acc1.mean(), original_mean);
}

// Test empty accumulator statistics
TEST_F(WelfordAccumulatorTest, EmptyAccumulatorStatistics) {
    welford_accumulator<double> acc;

    EXPECT_TRUE(acc.empty());
    EXPECT_EQ(acc.size(), 0u);
    EXPECT_EQ(acc.mean(), 0.0);
    EXPECT_EQ(acc.sum(), 0.0);
    EXPECT_EQ(acc.variance(), 0.0);
    EXPECT_EQ(acc.sample_variance(), 0.0);
    EXPECT_EQ(acc.std_dev(), 0.0);
    EXPECT_EQ(acc.sample_std_dev(), 0.0);
}
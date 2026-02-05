/**
 * @file test_extended_accumulators.cpp
 * @brief Tests for extended accumulators (EMA, covariance, histogram, quantile)
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <numeric>

#include "../include/accumux/accumulators/ema.hpp"
#include "../include/accumux/accumulators/covariance.hpp"
#include "../include/accumux/accumulators/histogram.hpp"
#include "../include/accumux/accumulators/quantile.hpp"
#include "../include/accumux/accumulators/welford.hpp"
#include "../include/accumux/accumulators/basic.hpp"
#include "../include/accumux/core/composition.hpp"

using namespace accumux;

// ============================================================================
// EMA Accumulator Tests
// ============================================================================

class EMATest : public ::testing::Test {
protected:
    static constexpr double EPSILON = 1e-10;
};

TEST_F(EMATest, DefaultConstruction) {
    ema_accumulator<double> ema;
    EXPECT_TRUE(ema.empty());
    EXPECT_EQ(ema.size(), 0);
}

TEST_F(EMATest, BasicAccumulation) {
    ema_accumulator<double> ema(0.5);  // High alpha for quick response

    ema += 10.0;
    EXPECT_NEAR(ema.eval(), 10.0, EPSILON);  // First value is the EMA

    ema += 20.0;
    // EMA = 0.5 * 20 + 0.5 * 10 = 15
    EXPECT_NEAR(ema.eval(), 15.0, EPSILON);
}

TEST_F(EMATest, FromPeriod) {
    auto ema = ema_accumulator<double>::from_period(10);
    // alpha = 2/(10+1) = 2/11 ≈ 0.1818
    EXPECT_NEAR(ema.alpha(), 2.0/11.0, EPSILON);
}

TEST_F(EMATest, FromHalfLife) {
    auto ema = ema_accumulator<double>::from_half_life(5.0);
    // alpha = 1 - exp(-ln(2)/5)
    double expected_alpha = 1.0 - std::exp(-std::log(2.0) / 5.0);
    EXPECT_NEAR(ema.alpha(), expected_alpha, EPSILON);
}

TEST_F(EMATest, InvalidAlpha) {
    EXPECT_THROW(ema_accumulator<double>(0.0), std::invalid_argument);
    EXPECT_THROW(ema_accumulator<double>(-0.5), std::invalid_argument);
    EXPECT_THROW(ema_accumulator<double>(1.5), std::invalid_argument);
}

TEST_F(EMATest, ConceptCompliance) {
    static_assert(Accumulator<ema_accumulator<double>>);
    static_assert(StatisticalAccumulator<ema_accumulator<double>>);
}

// ============================================================================
// Covariance Accumulator Tests
// ============================================================================

class CovarianceTest : public ::testing::Test {
protected:
    static constexpr double EPSILON = 1e-10;
};

TEST_F(CovarianceTest, DefaultConstruction) {
    covariance_accumulator<double> cov;
    EXPECT_TRUE(cov.empty());
    EXPECT_EQ(cov.size(), 0);
}

TEST_F(CovarianceTest, PerfectPositiveCorrelation) {
    covariance_accumulator<double> cov;

    // y = x (perfect positive correlation)
    for (int i = 1; i <= 10; ++i) {
        cov += std::make_pair(static_cast<double>(i), static_cast<double>(i));
    }

    EXPECT_NEAR(cov.correlation(), 1.0, EPSILON);
}

TEST_F(CovarianceTest, PerfectNegativeCorrelation) {
    covariance_accumulator<double> cov;

    // y = -x (perfect negative correlation)
    for (int i = 1; i <= 10; ++i) {
        cov += std::make_pair(static_cast<double>(i), static_cast<double>(-i));
    }

    EXPECT_NEAR(cov.correlation(), -1.0, EPSILON);
}

TEST_F(CovarianceTest, LinearRegression) {
    covariance_accumulator<double> cov;

    // y = 2x + 3
    for (int i = 0; i < 10; ++i) {
        double x = static_cast<double>(i);
        double y = 2.0 * x + 3.0;
        cov += std::make_pair(x, y);
    }

    EXPECT_NEAR(cov.slope(), 2.0, EPSILON);
    EXPECT_NEAR(cov.intercept(), 3.0, EPSILON);
    EXPECT_NEAR(cov.r_squared(), 1.0, EPSILON);
}

TEST_F(CovarianceTest, MeanCalculation) {
    covariance_accumulator<double> cov;

    cov += std::make_pair(1.0, 10.0);
    cov += std::make_pair(2.0, 20.0);
    cov += std::make_pair(3.0, 30.0);

    EXPECT_NEAR(cov.mean_x(), 2.0, EPSILON);
    EXPECT_NEAR(cov.mean_y(), 20.0, EPSILON);
}

TEST_F(CovarianceTest, CombineAccumulators) {
    covariance_accumulator<double> cov1, cov2;

    cov1 += std::make_pair(1.0, 2.0);
    cov1 += std::make_pair(2.0, 4.0);

    cov2 += std::make_pair(3.0, 6.0);
    cov2 += std::make_pair(4.0, 8.0);

    cov1 += cov2;

    EXPECT_EQ(cov1.size(), 4);
    EXPECT_NEAR(cov1.correlation(), 1.0, EPSILON);
}

TEST_F(CovarianceTest, ConceptCompliance) {
    static_assert(Accumulator<covariance_accumulator<double>>);
}

// ============================================================================
// Histogram Accumulator Tests
// ============================================================================

class HistogramTest : public ::testing::Test {
protected:
    static constexpr double EPSILON = 1e-10;
};

TEST_F(HistogramTest, BasicConstruction) {
    histogram_accumulator<double> hist(0.0, 10.0, 10);

    EXPECT_EQ(hist.num_bins(), 10);
    EXPECT_NEAR(hist.min(), 0.0, EPSILON);
    EXPECT_NEAR(hist.max(), 10.0, EPSILON);
    EXPECT_NEAR(hist.bin_width(), 1.0, EPSILON);
}

TEST_F(HistogramTest, BinPlacement) {
    histogram_accumulator<double> hist(0.0, 10.0, 10);

    hist += 0.5;  // Should go in bin 0
    hist += 5.5;  // Should go in bin 5
    hist += 9.5;  // Should go in bin 9

    EXPECT_EQ(hist.bin_count(0), 1);
    EXPECT_EQ(hist.bin_count(5), 1);
    EXPECT_EQ(hist.bin_count(9), 1);
    EXPECT_EQ(hist.total(), 3);
}

TEST_F(HistogramTest, UnderflowOverflow) {
    histogram_accumulator<double> hist(0.0, 10.0, 10);

    hist += -5.0;  // Underflow
    hist += 15.0;  // Overflow
    hist += 5.0;   // Normal

    EXPECT_EQ(hist.underflow(), 1);
    EXPECT_EQ(hist.overflow(), 1);
    EXPECT_EQ(hist.total(), 3);
}

TEST_F(HistogramTest, QuantileEstimation) {
    histogram_accumulator<double> hist(0.0, 100.0, 100);

    // Add 100 values from 0 to 99
    for (int i = 0; i < 100; ++i) {
        hist += static_cast<double>(i);
    }

    // Median should be around 49-50
    double median = hist.median();
    EXPECT_NEAR(median, 50.0, 2.0);  // Allow some error due to binning
}

TEST_F(HistogramTest, CombineHistograms) {
    histogram_accumulator<double> hist1(0.0, 10.0, 10);
    histogram_accumulator<double> hist2(0.0, 10.0, 10);

    hist1 += 1.5;
    hist1 += 2.5;

    hist2 += 3.5;
    hist2 += 4.5;

    hist1 += hist2;

    EXPECT_EQ(hist1.total(), 4);
    EXPECT_EQ(hist1.bin_count(1), 1);
    EXPECT_EQ(hist1.bin_count(2), 1);
    EXPECT_EQ(hist1.bin_count(3), 1);
    EXPECT_EQ(hist1.bin_count(4), 1);
}

TEST_F(HistogramTest, InvalidConstruction) {
    EXPECT_THROW(histogram_accumulator<double>(10.0, 0.0, 10), std::invalid_argument);
    EXPECT_THROW(histogram_accumulator<double>(0.0, 10.0, 0), std::invalid_argument);
}

TEST_F(HistogramTest, ConceptCompliance) {
    static_assert(Accumulator<histogram_accumulator<double>>);
}

// ============================================================================
// P2 Quantile Accumulator Tests
// ============================================================================

class P2QuantileTest : public ::testing::Test {
protected:
    static constexpr double EPSILON = 0.1;  // P2 is approximate
};

TEST_F(P2QuantileTest, MedianEstimation) {
    p2_quantile_accumulator<double> median(0.5);

    // Add 1000 values from 0 to 999
    for (int i = 0; i < 1000; ++i) {
        median += static_cast<double>(i);
    }

    // True median is 499.5
    EXPECT_NEAR(median.eval(), 499.5, 50.0);  // Allow 10% error
}

TEST_F(P2QuantileTest, PercentileEstimation) {
    p2_quantile_accumulator<double> p90(0.9);

    for (int i = 0; i < 1000; ++i) {
        p90 += static_cast<double>(i);
    }

    // 90th percentile should be around 900
    EXPECT_NEAR(p90.eval(), 900.0, 100.0);
}

TEST_F(P2QuantileTest, InvalidQuantile) {
    EXPECT_THROW(p2_quantile_accumulator<double>(0.0), std::invalid_argument);
    EXPECT_THROW(p2_quantile_accumulator<double>(1.0), std::invalid_argument);
    EXPECT_THROW(p2_quantile_accumulator<double>(-0.5), std::invalid_argument);
}

TEST_F(P2QuantileTest, SmallSample) {
    p2_quantile_accumulator<double> median(0.5);

    median += 1.0;
    median += 2.0;
    median += 3.0;

    // Should return something reasonable even with < 5 samples
    double result = median.eval();
    EXPECT_GE(result, 1.0);
    EXPECT_LE(result, 3.0);
}

TEST_F(P2QuantileTest, ConceptCompliance) {
    static_assert(Accumulator<p2_quantile_accumulator<double>>);
}

// ============================================================================
// Reservoir Quantile Accumulator Tests
// ============================================================================

class ReservoirQuantileTest : public ::testing::Test {
protected:
    static constexpr double EPSILON = 1e-10;
};

TEST_F(ReservoirQuantileTest, ExactMedianSmallSample) {
    reservoir_quantile_accumulator<double> rq(100);

    rq += 1.0;
    rq += 2.0;
    rq += 3.0;
    rq += 4.0;
    rq += 5.0;

    EXPECT_NEAR(rq.median(), 3.0, EPSILON);
}

TEST_F(ReservoirQuantileTest, Quartiles) {
    reservoir_quantile_accumulator<double> rq(1000);

    // Add values 1-100
    for (int i = 1; i <= 100; ++i) {
        rq += static_cast<double>(i);
    }

    EXPECT_NEAR(rq.q1(), 25.75, 1.0);
    EXPECT_NEAR(rq.median(), 50.5, 1.0);
    EXPECT_NEAR(rq.q3(), 75.25, 1.0);
}

TEST_F(ReservoirQuantileTest, IQR) {
    reservoir_quantile_accumulator<double> rq(1000);

    for (int i = 1; i <= 100; ++i) {
        rq += static_cast<double>(i);
    }

    double iqr = rq.iqr();
    EXPECT_NEAR(iqr, 49.5, 2.0);  // Q3 - Q1
}

TEST_F(ReservoirQuantileTest, MultipleQuantiles) {
    reservoir_quantile_accumulator<double> rq(1000);

    for (int i = 0; i < 100; ++i) {
        rq += static_cast<double>(i);
    }

    auto quantiles = rq.quantiles({0.1, 0.5, 0.9});
    EXPECT_EQ(quantiles.size(), 3);
}

TEST_F(ReservoirQuantileTest, ConceptCompliance) {
    static_assert(Accumulator<reservoir_quantile_accumulator<double>>);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(ExtendedAccumulatorIntegration, CompositionWithEMA) {
    auto stats = ema_accumulator<double>(0.1) + welford_accumulator<double>();

    for (int i = 1; i <= 100; ++i) {
        stats += static_cast<double>(i);
    }

    auto ema = stats.get_first();
    auto welford = stats.get_second();

    EXPECT_FALSE(ema.empty());
    EXPECT_EQ(welford.size(), 100);
}

TEST(ExtendedAccumulatorIntegration, HistogramWithMinMax) {
    auto stats = histogram_accumulator<double>(0.0, 100.0, 10) + minmax_accumulator<double>();

    std::vector<double> data = {5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0};
    for (double v : data) {
        stats += v;
    }

    auto hist = stats.get_first();
    auto mm = stats.get_second();

    EXPECT_EQ(hist.total(), 10);
    EXPECT_NEAR(mm.min(), 5.0, 1e-10);
    EXPECT_NEAR(mm.max(), 95.0, 1e-10);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

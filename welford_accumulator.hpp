#pragma once

#include "kbn_sum.hpp"

namespace algebraic_accumulators
{
    template <typename A>
    struct welford_accumulator
    {
        using value_type = typename A::value_type;

        welford_accumulator() : count(0), mu(), M2() {}
        
        explicit welford_accumulator(value_type a) : count(0), mu(), M2()
            { operator+=(a); }

        explicit welford_accumulator(welford_accumulator const &) = default;

        auto variance() const { return value_type(M2) / value_type(count); }
        auto sample_variance() const { return value_type(M2) / (value_type(count - 1)); }
        auto mean() const { return value_type(mu); }
        auto sum() const { return value_type(mu) * value_type(count); }
        auto size() const { return count; }
        operator value_type() const { return sum(); }

        auto & operator+=(value_type x)
        {
            ++count;
            auto delta = x - mu;
            mu += delta / count;
            auto delta2 = std::move(x) - mu;
            M2 += delta * delta2;
            return *this;
        }

        unsigned int count;
        A mu, M2;
    };

    template <typename T>
    using kbn_welford_accumulate = welford_accumulator<kbn_sum<T>>;

    template <typename A>
    auto mean(welford_accumulator<A> const & acc)
    {
        return acc.mean();
    }

    template <typename A>
    auto variance(welford_accumulator<A> const & acc)
    {
        return acc.variance();
    }

    template <typename A>
    auto sample_variance(welford_accumulator<A> const & acc)
    {
        return acc.sample_variance();
    }

    template <typename A>
    auto size(welford_accumulator<A> const & acc)
    {
        return acc.size();
    }

    template <typename A>
    auto sum(welford_accumulator<A> const & acc)
    {
        return acc.sum();
    }

}
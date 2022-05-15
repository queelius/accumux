#pragma once

#include "accumulator_exp.hpp"

namespace algebraic_accumulators
{
    template <typename A, typename B, typename F>
    struct binary_accumulator_exp: accumulator_exp<binary_accumulator_exp<A,B,F>>
    {
        auto eval() const { return f(left,right); }

        F f;
        A const & left;
        B const & right;
    };

    template <typename A, typename B>
    auto left(binary_accumulator_exp<A,B> const & acc)
    {
        return acc.left;
    }

    template <typename A, typename B>
    auto right(binary_accumulator_exp<A,B> const & acc)
    {
        return acc.right;
    }

}
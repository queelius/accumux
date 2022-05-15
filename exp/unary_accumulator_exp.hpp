#pragma once

#include "accumulator_exp.hpp"

namespace algebraic_accumulators
{
    template <typename A, typename F>
    struct unary_accumulator_exp: accumulator_exp<unary_accumulator_exp<A,F>>
    {
        auto eval() const { return f(acc); }

        F f;
        A const & acc;
    };
}
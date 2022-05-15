


#pragma once

template <typename E>
struct accumulator_exp
{
    auto eval() const { static_cast<E const &>(e).eval(); }

    E const & e;
}
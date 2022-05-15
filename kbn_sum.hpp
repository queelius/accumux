#pragma once

/**
 * Given a data type X that models SumReduce<T>, X must be a RegularType
 * and a monoid over (X, +, X()) with two homomorphisms to and from the
 * monoid (T, +, T(0)).
 * 
 * To construct the identity value under + : (X,X) -> X, use the default
 * constructor
 *     X::X().
 *
 * To map X to T, apply the conversion operator
 *     operator X::T().
 *
 * To map T to X, apply the value constructor
 *     X::X(T const & t).
 *
 * To be a homomorphism, the following must be true:
 *     X(T(x) + T(y)) = X(T(x)) + X(T(y)).
 *
 * Due to numerical errors, this may only be *approximately* true,
 * especially since X, as a model of SumReduce<T>, will generally have
 * smaller rounding errors on + : X^* -> X than + : T^* -> T.
 * (X will generlally more accurately represent the concept that T itself
 * models over +.)
 * 
 * In the context of kbn_sum<T>,
 *         it only makes sense for T to be a type that approximately models
 *         the concept of some other type, like a floating point type
 *         modeling real numbers.
 *
 * Since the SumReduce<T> concept is normally used to efficiently
 * reduce sequences over +, there is a specially defined procedure to
 * perform this reduction:
 *
 *     template <InputIterator I>
 *     X::reduce(I, I) -> &X.
 *
 * Typically, this reduction has a time complexity of O(n) and a space
 * complexity of O(1), where n is the number of elements being reduced. Some
 * reductions may have higher or even lower complexities.
 * 
 * If the reduction takes the sequence and re-orders them in a way to mitigate
 * accumulation error, then the time complexity may be O(n log n) for sorts
 * based on binary comparisons. If the reduction employs random sampling
 * and I models a random access iterator, then the reduction has a time
 * complexity on the order of the size of the sample, but generally has a
 * greater loss in accuracy than naively summing over T.
 */

#include <algorithm>
#include "exp/accumulator_exp.hpp"
using std::abs;

namespace algebraic_accumulator
{
    /**
     * kbn_sum<T> models the SumReduce<T> concept with space complexity O(1)
     * and time complexity O(n).
     *
     * kbn_sum<T> has the additional following requirements on T:
     *
     *     (2) abs : T -> T models an infinity norm.
     *
     * We do not really need a vector space, since kbn_sum<T> does not
     * need scalar multiplication.
     *
     * A specialization might be given for kbn_sum<V> where V is a vector space
     * over T of dimension greater than 2 by modeling + : (V,V) -> V by
     * doing a kbn_sum<T> over each component separately. However, we
     * prefer to employ the underlying + : (V,V) -> V operator, which might
     * be highly optimized.
     */
    template <typename T>
    struct kbn_sum
    {
        using value_type = T;

        kbn_sum(T const & x = T(0)) : s(x), c(T(0)) {};

        kbn_sum(kbn_sum<T> const &) = default;
        kbn_sum<T> & operator=(kbn_sum<T> const &) = default;
        auto & operator=(T const & x) { s = x; return *this; }
        operator T() const { return s+c; }
        auto eval() const { return s+c; }

        auto operator<(T const & rhs) const { return (T)(*this) < rhs; }

        auto operator<(kbn_sum<T> const & rhs) const
        {
            return (T)(*this) < (T)rhs;
        }

        /**
         * If a value is the result of a total obtained
         * by 'naively' summing a sequence of k values of type
         * T, then the numerical error (approximation error)
         * in worst-case is O(k).
         *
         * kbn_sum<T> is designed explicitly to reduce the
         * numerical error from summation with a type T that
         * has fewer digits of precision than the type R that it
         * models. In particular, the approximation error is
         * independent of k and is bounded by the precision of
         * T.
         */
        auto operator==(kbn_sum<T> const & rhs) const
        {
            return (T)(*this) == (T)rhs;
        }

        auto operator+(kbn_sum<T> const & rhs)
        {
            return kbn_sum<T>(*this) += rhs;
        }

        template <typename I>
        auto & reduce(I begin, I end)
        {
            for (auto i = begin; i != end; ++i)
                (*this) += *i;
            return *this;
        }

        auto & operator+=(kbn_sum<T> const & rhs)
        {
            (*this) += rhs.s;
            (*this) += rhs.c;
            return *this;
        }

        auto & operator+=(T const & x)
        {
            auto t = s + x;
            if (abs(x) < abs(s))
                c += ((s - t) + x);
            else
                c += ((x - t) + s);
            s = t;
            return *this;
        }

        T s, c;
    };

    template <typename T>
    auto abs(kbn_sum<T> x)
    {
        x.s = abs(x.s);
        x.c = abs(x.c);
        return x;
    }
}

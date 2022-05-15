# Specialized reductions (folds) on sequences

## Introduction

A reducer is a a device that reduces a large data set into a small
data set. The most common example follows the pattern of applying a binary
operator to the data set such that we reduce a sequence of values of type `T`
to a single value of type `T`, such as taking the sum or mean of a data set.

Other names for this type of reduction are `fold` and `accumulate`.

Example:

Given the sequence `{1,2,3}` and the binary operation `+`, `{1,2,3}` is reduced
to `6`.

Since `+` is a binary operation, it requires two values, and so may view the
reduction as the composition

    +(1,+(2,3)) => 6.

However, what is the reduction of `{1}`? Since `+` is a binary operator, we also
allow for an *initial* value `c`, such that the reduction of `{1}` is given by

    +(1,c).

By default, the initial value should be the identity element for the operation,
e.g., the identity element for `+` is `0`, thus the reduction of `{1}` with `+`
is `+(1,0)` and the reduction of `{1,2,3}` with `+` is given by

    +(1,+(2,+(3,0))) => 6,

the same as before. Note that, by the associativity of `+`, we could rewrite the
above as

    +(0,+(3,+(2,1))) => 6.

This is known as a left reduce or left fold, and the original composition is
known as a right reduce or right fold.

If the operation is associative and commutative, then the two variations are
mathematically equivalent, but there may be computational efficiencies
to be had for using one over the other depending upon the nature of the
reduction.

Finally, we have one more case to consider.
What is the reduction of `{}`? Since we only have one value, the initial value,
the binary operation cannot be applied.
In this case, we simply return the initial
value, i.e., `+` on `{}` using initial value `c` is `c`.

The operation is typically associative.

## Programmatic interface

A reducer `A` should be a regular type with overloads of the form:

```cpp
    template <reducer A>
    A::value_type -> T

    template <reducer A>
    operator += (A&,T) -> A&

    template <reducer A>
    = : (A&,T) -> A&

    template <reducer A>
    A() -> T

    template <reducer A, forward_iterator I>
    A(I,I,T) -> A&

    template <reducer A, forward_iterator I>
    A(I,I) -> A&
```

so that a sequence of elements `{x[t]}` may be reduced with:

```cpp
    A res = init;
    for (auto e : x)
        res += e;

    A res2 = init;
    res2(begin(x),end(x));
    assert(res2()==res());

    A res3(begin(x),end(x),init);
    assert(res3()==res2());

```

## Specializations

If we have a summation reducer `r` of type `A`, then we may obtain its
current value (sum) with `r()`.

If we have the number of elements that were summed in the reducer, say `n`,
then the mean of the reduced sequence is given by `r() / n`.
Thus, a reducer that models the concept of summation may be used to
compute, say, the mean or variance.
However, there are specialized procedures that may be used to more efficientl
(say, single pass) and accurately compute these results, such as the Wellford
reducer, which computes both the mean and variance of a sequence.

If `A` may be used to more efficiently compute the mean or variance (or both),
then it should override the default implementations.

What if we have a set of reducers? If we wish to fine the mean over the
mean of the reducers, we could compute the sum of the means and then take
the mean of the sum.

However, there may be more efficient and more accurate ways.

In what follows, we consider an *algebra* over reducers.
Consider the operation

```cpp
    template <typename A, typename B>
    + : (reducer_exp<A>,reducer_exp<B>) ->
        sum_reducer_exp<...>
```

where `sum_reducer_exp` models the concept of the sum of these two reducers. However, since we generally prefer expressions to be constant,
`sum_reducer_exp` is not itself an reducer, but an reducer
expression.

As an expression, we may perform additional operations on it, such as
the mean operation:

```cpp
    template <typename A, typename B>
    mean : sum_reducer_exp<...> -> mean_reducer_exp<...>
```

may be used to efficiently (and lazily) compute the mean of these reducers.

However, for 




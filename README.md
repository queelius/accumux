# Algebraic accumulators

Definition: an accumulator is a a device for the purpose of (efficiently)
computing the application of some function to a sequence of data.

An accumulator `A` should be a regular type (with a default constructor that
computes the additive identity) with overloads of the form:

```cpp
    template <accumulator A>
    += : (A&,value_type<A>) -> A&

    template <accumulator A>
    = : (A&,value_type<A>) -> A&

    template <accumulator A>
    A : () -> value_type<A>

    template <accumulator A>
    value_type : A -> type
```

so that a sequence of elements `{x[t]}` may be *accumulated* with:

```cpp
    A acc = init; // A acc by itself constructs the additive identity
    for (auto e : x)
        acc += e;
    return result;
```

Alternatively, using an iterator approach, the same result is obtained with
`A(begin(x),end(x),init)`.

The operation performed by `A` is an associative operation, e.g., the
canonical example is standard addition, but minimum and other kinds of
operations may also be used.

## Algebraic properties

If we have a summation accumulator `acc` of type `A`, then we may obtain its accumulated results with `acc()`.

If we have the number of elements that were summed in the accumulator, say `n`,
then `mean(acc) := acc() / n`.
Thus, any accumulator that models the concept of summation may be used to
compute, say, the mean or variance.
However, there are specialized procedures that may be used to more efficientl
(say, single pass) and accurately compute these results, such as the Wellford
accumulator, which computes both the mean and variance of a sequence.

If `A` may be used to more efficiently compute the mean or variance (or both),
then it should override the default implementations.

What if we have a set of accumulators? If we wish to fine the mean over the
mean of the accumulators, we could compute the sum of the means and then take
the mean of the sum.

However, there may be more efficient and more accurate ways.

In what follows, we consider an *algebra* over accumulators.
Consider the operation

```cpp
    template <typename A, typename B>
    + : (accumulator_exp<A>,accumulator_exp<B>) ->
        sum_accumulator_exp<...>
```

where `sum_accumulator_exp` models the concept of the sum of these two accumulators. However, since we generally prefer expressions to be constant,
`sum_accumulator_exp` is not itself an accumulator, but an accumulator
expression.

As an expression, we may perform additional operations on it, such as
the mean operation:

```cpp
    template <typename A, typename B>
    mean : sum_accumulator_exp<...> -> mean_accumulator_exp<...>
```

may be used to efficiently (and lazily) compute the mean of these accumulators.

However, for 




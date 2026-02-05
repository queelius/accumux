/**
 * @file variadic_composition.hpp
 * @brief Variadic parallel composition with fold expressions
 *
 * Enables composing arbitrary numbers of accumulators with zero overhead
 * using C++17/20 fold expressions and parameter packs.
 */

#pragma once

#include "accumulator_concept.hpp"
#include <tuple>
#include <utility>
#include <type_traits>

namespace accumux {

/**
 * @brief Variadic parallel composition - N accumulators process same data
 *
 * This is more efficient than nested binary compositions for large N,
 * as it avoids deep template instantiation and provides a flat tuple result.
 *
 * @tparam Accums Pack of accumulator types
 */
template<Accumulator... Accums>
class variadic_parallel_composition {
public:
    using value_type = std::tuple<typename Accums::value_type...>;

private:
    std::tuple<Accums...> accumulators_;

    // Helper to apply operation to all accumulators
    template<typename F, std::size_t... Is>
    void apply_all(F&& f, std::index_sequence<Is...>) {
        (f(std::get<Is>(accumulators_)), ...);
    }

    template<typename F, std::size_t... Is>
    void apply_all(F&& f, std::index_sequence<Is...>) const {
        (f(std::get<Is>(accumulators_)), ...);
    }

public:
    static constexpr std::size_t accumulator_count = sizeof...(Accums);

    /**
     * @brief Default constructor
     */
    variadic_parallel_composition() = default;

    /**
     * @brief Construct with initial accumulators
     */
    explicit variadic_parallel_composition(Accums... accs)
        : accumulators_(std::move(accs)...) {}

    /**
     * @brief Construct from tuple
     */
    explicit variadic_parallel_composition(std::tuple<Accums...> accs)
        : accumulators_(std::move(accs)) {}

    variadic_parallel_composition(const variadic_parallel_composition&) = default;
    variadic_parallel_composition(variadic_parallel_composition&&) = default;
    variadic_parallel_composition& operator=(const variadic_parallel_composition&) = default;
    variadic_parallel_composition& operator=(variadic_parallel_composition&&) = default;

    /**
     * @brief Add value to all accumulators using fold expression
     */
    template<typename T>
    variadic_parallel_composition& operator+=(const T& value) {
        std::apply([&value](auto&... accs) {
            (void(accs += value), ...);  // Fold expression
        }, accumulators_);
        return *this;
    }

    /**
     * @brief Combine with another variadic composition
     */
    variadic_parallel_composition& operator+=(const variadic_parallel_composition& other) {
        combine_impl(other, std::index_sequence_for<Accums...>{});
        return *this;
    }

    /**
     * @brief Get result tuple
     */
    value_type eval() const {
        return std::apply([](const auto&... accs) {
            return std::make_tuple(accs.eval()...);
        }, accumulators_);
    }

    explicit operator value_type() const {
        return eval();
    }

    /**
     * @brief Get accumulator by index
     */
    template<std::size_t I>
    const auto& get() const {
        static_assert(I < sizeof...(Accums), "Index out of bounds");
        return std::get<I>(accumulators_);
    }

    template<std::size_t I>
    auto& get() {
        static_assert(I < sizeof...(Accums), "Index out of bounds");
        return std::get<I>(accumulators_);
    }

    /**
     * @brief Get accumulator by type (first match)
     */
    template<typename T>
    const T& get() const {
        return std::get<T>(accumulators_);
    }

    template<typename T>
    T& get() {
        return std::get<T>(accumulators_);
    }

    /**
     * @brief Get underlying tuple
     */
    const std::tuple<Accums...>& accumulators() const {
        return accumulators_;
    }

    /**
     * @brief Apply a function to each accumulator
     */
    template<typename F>
    void for_each(F&& f) {
        std::apply([&f](auto&... accs) {
            (f(accs), ...);
        }, accumulators_);
    }

    template<typename F>
    void for_each(F&& f) const {
        std::apply([&f](const auto&... accs) {
            (f(accs), ...);
        }, accumulators_);
    }

    /**
     * @brief Transform and collect results
     */
    template<typename F>
    auto transform(F&& f) const {
        return std::apply([&f](const auto&... accs) {
            return std::make_tuple(f(accs)...);
        }, accumulators_);
    }

private:
    template<std::size_t... Is>
    void combine_impl(const variadic_parallel_composition& other, std::index_sequence<Is...>) {
        ((std::get<Is>(accumulators_) += std::get<Is>(other.accumulators_)), ...);
    }
};

// Verify concept compliance
template<typename... Ts>
struct variadic_concept_check {
    static constexpr bool check() {
        if constexpr (sizeof...(Ts) > 0) {
            return (Accumulator<Ts> && ...);
        }
        return true;
    }
};

/**
 * @brief Factory function for variadic parallel composition
 */
template<Accumulator... Accums>
auto make_parallel(Accums&&... accums) {
    return variadic_parallel_composition<std::decay_t<Accums>...>(
        std::forward<Accums>(accums)...
    );
}

/**
 * @brief Concatenate two variadic compositions
 */
template<Accumulator... As, Accumulator... Bs>
auto concat(const variadic_parallel_composition<As...>& a,
            const variadic_parallel_composition<Bs...>& b) {
    return std::apply([&b](const auto&... as) {
        return std::apply([&as...](const auto&... bs) {
            return variadic_parallel_composition<As..., Bs...>(as..., bs...);
        }, b.accumulators());
    }, a.accumulators());
}

/**
 * @brief Create composition from tuple of accumulators
 */
template<Accumulator... Accums>
auto from_tuple(std::tuple<Accums...> accs) {
    return variadic_parallel_composition<Accums...>(std::move(accs));
}

/**
 * @brief Helper to create N copies of same accumulator type
 */
template<std::size_t N, Accumulator Acc>
auto replicate() {
    return []<std::size_t... Is>(std::index_sequence<Is...>) {
        return variadic_parallel_composition<decltype((void(Is), Acc{}))...>(
            ((void)Is, Acc{})...
        );
    }(std::make_index_sequence<N>{});
}

// ADL-enabled get for structured bindings
template<std::size_t I, Accumulator... Accums>
const auto& get(const variadic_parallel_composition<Accums...>& comp) {
    return comp.template get<I>();
}

template<std::size_t I, Accumulator... Accums>
auto& get(variadic_parallel_composition<Accums...>& comp) {
    return comp.template get<I>();
}

} // namespace accumux

// Structured binding support
namespace std {
    template<typename... Accums>
    struct tuple_size<accumux::variadic_parallel_composition<Accums...>>
        : std::integral_constant<std::size_t, sizeof...(Accums)> {};

    template<std::size_t I, typename... Accums>
    struct tuple_element<I, accumux::variadic_parallel_composition<Accums...>> {
        using type = std::tuple_element_t<I, std::tuple<Accums...>>;
    };
}

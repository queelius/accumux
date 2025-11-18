/**
 * @file composition.hpp
 * @brief Core composition operations for accumux accumulators
 * 
 * This file defines the basic algebraic operations for composing accumulators:
 * - Parallel composition (a + b): Both accumulators process the same data
 * - Sequential composition (a * b): Output of a becomes input to b
 * - Conditional composition (a | b): Choose based on conditions
 * 
 * @version 1.0.0
 * @date 2025
 * @copyright MIT License
 */

#pragma once

#include <tuple>
#include <variant>
#include <type_traits>
#include <chrono>
#include <numeric>
#include "accumulator_concept.hpp"

namespace accumux {

/**
 * @brief Parallel composition - both accumulators process the same data stream
 * 
 * This allows computing multiple statistics in a single pass over the data.
 * Mathematical interpretation: (a + b)(x) = a(x) ⊕ b(x)
 * 
 * @tparam AccumA First accumulator type
 * @tparam AccumB Second accumulator type
 */
template<Accumulator AccumA, Accumulator AccumB>
class parallel_composition {
public:
    // Result type is a tuple of both accumulator results
    using value_type = std::tuple<typename AccumA::value_type, typename AccumB::value_type>;

private:
    AccumA accumulator_a_;
    AccumB accumulator_b_;

public:
    /**
     * @brief Default constructor - initialize both accumulators
     */
    parallel_composition() = default;

    /**
     * @brief Construct with initial accumulators
     */
    parallel_composition(AccumA a, AccumB b) : accumulator_a_(std::move(a)), accumulator_b_(std::move(b)) {}

    /**
     * @brief Add a value to both accumulators
     */
    template<typename T>
    parallel_composition& operator+=(const T& value) {
        accumulator_a_ += value;
        accumulator_b_ += value;
        return *this;
    }
    
    /**
     * @brief Combine with another parallel composition
     */
    parallel_composition& operator+=(const parallel_composition& other) {
        accumulator_a_ += other.accumulator_a_;
        accumulator_b_ += other.accumulator_b_;
        return *this;
    }
    
    /**
     * @brief Get the first accumulator (by type)
     */
    template<typename T>
    const T& get() const {
        if constexpr (std::is_same_v<T, AccumA>) {
            return accumulator_a_;
        } else if constexpr (std::is_same_v<T, AccumB>) {
            return accumulator_b_;
        } else {
            static_assert(std::is_same_v<T, AccumA> || std::is_same_v<T, AccumB>, 
                         "Requested type not in composition");
        }
    }
    
    /**
     * @brief Get first accumulator by index
     */
    const AccumA& get_first() const { return accumulator_a_; }
    
    /**
     * @brief Get second accumulator by index
     */
    const AccumB& get_second() const { return accumulator_b_; }
    
    /**
     * @brief Evaluate the composition (returns tuple of results)
     */
    value_type eval() const {
        return std::make_tuple(accumulator_a_.eval(), accumulator_b_.eval());
    }

    /**
     * @brief Explicit conversion to result tuple
     */
    explicit operator value_type() const {
        return eval();
    }
};

/**
 * @brief Sequential composition - pipeline one accumulator into another
 * 
 * The output of the first accumulator becomes input to the second.
 * Mathematical interpretation: (a * b)(x) = b(a(x))
 * 
 * @tparam AccumA First accumulator (data processor)
 * @tparam AccumB Second accumulator (result processor)
 */
template<Accumulator AccumA, Accumulator AccumB>
class sequential_composition {
public:
    using value_type = typename AccumB::value_type;
    
private:
    AccumA accumulator_a_;
    AccumB accumulator_b_;
    
public:
    sequential_composition() = default;
    sequential_composition(AccumA a, AccumB b) : accumulator_a_(std::move(a)), accumulator_b_(std::move(b)) {}
    
    /**
     * @brief Process value through the pipeline
     */
    template<typename T>
    sequential_composition& operator+=(T&& value) {
        // Add to first accumulator
        accumulator_a_ += std::forward<T>(value);
        
        // Feed result to second accumulator
        // Note: This is conceptual - real implementation depends on accumulator semantics
        accumulator_b_ += accumulator_a_.eval();
        
        return *this;
    }
    
    /**
     * @brief Combine with another sequential composition
     */
    sequential_composition& operator+=(const sequential_composition& other) {
        accumulator_a_ += other.accumulator_a_;
        accumulator_b_ += other.accumulator_b_;
        return *this;
    }

    /**
     * @brief Evaluate the final result
     */
    value_type eval() const {
        return accumulator_b_.eval();
    }

    /**
     * @brief Explicit conversion operator
     */
    explicit operator value_type() const {
        return eval();
    }

    /**
     * @brief Get intermediate result from first accumulator
     */
    auto intermediate() const {
        return accumulator_a_.eval();
    }
};

/**
 * @brief Conditional composition - choose accumulator based on predicate
 * 
 * @tparam AccumA First accumulator option
 * @tparam AccumB Second accumulator option  
 * @tparam Predicate Function type that determines which accumulator to use
 */
template<typename AccumA, typename AccumB, typename Predicate>
class conditional_composition {
public:
    using value_type = std::common_type_t<typename AccumA::value_type, typename AccumB::value_type>;

private:
    std::variant<AccumA, AccumB> active_accumulator_;
    Predicate predicate_;

public:
    conditional_composition() = default;

    conditional_composition(AccumA a, AccumB b, Predicate pred)
        : active_accumulator_(std::move(a)), predicate_(std::move(pred)) {}

    conditional_composition(const conditional_composition&) = default;
    conditional_composition(conditional_composition&&) = default;
    conditional_composition& operator=(const conditional_composition&) = default;
    conditional_composition& operator=(conditional_composition&&) = default;

    template<typename T>
    conditional_composition& operator+=(T&& value) {
        // Switch accumulator if predicate changes
        if (predicate_(std::forward<T>(value))) {
            if (std::holds_alternative<AccumB>(active_accumulator_)) {
                active_accumulator_ = AccumA{};
            }
            std::get<AccumA>(active_accumulator_) += std::forward<T>(value);
        } else {
            if (std::holds_alternative<AccumA>(active_accumulator_)) {
                active_accumulator_ = AccumB{};
            }
            std::get<AccumB>(active_accumulator_) += std::forward<T>(value);
        }
        return *this;
    }

    conditional_composition& operator+=(const conditional_composition& other) {
        // Combine the active accumulators
        std::visit([this](const auto& other_acc) {
            std::visit([&other_acc](auto& this_acc) {
                using ThisType = std::decay_t<decltype(this_acc)>;
                using OtherType = std::decay_t<decltype(other_acc)>;
                if constexpr (std::is_same_v<ThisType, OtherType>) {
                    this_acc += other_acc;
                }
            }, this->active_accumulator_);
        }, other.active_accumulator_);
        return *this;
    }

    value_type eval() const {
        return std::visit([](const auto& acc) -> value_type {
            return static_cast<value_type>(acc.eval());
        }, active_accumulator_);
    }

    explicit operator value_type() const {
        return eval();
    }
};

// ============================================================================
// Operator Overloads for Natural Syntax
// ============================================================================

/**
 * @brief Parallel composition operator (a + b)
 */
template<Accumulator AccumA, Accumulator AccumB>
auto operator+(AccumA&& a, AccumB&& b) {
    return parallel_composition<std::decay_t<AccumA>, std::decay_t<AccumB>>(
        std::forward<AccumA>(a), std::forward<AccumB>(b)
    );
}

/**
 * @brief Sequential composition operator (a * b)
 */
template<Accumulator AccumA, Accumulator AccumB>
auto operator*(AccumA&& a, AccumB&& b) {
    return sequential_composition<std::decay_t<AccumA>, std::decay_t<AccumB>>(
        std::forward<AccumA>(a), std::forward<AccumB>(b)
    );
}

/**
 * @brief Conditional composition function (alternative to operator|)
 */
template<typename AccumA, typename AccumB, typename Predicate>
auto conditional(AccumA&& a, AccumB&& b, Predicate&& pred) {
    return conditional_composition<std::decay_t<AccumA>, std::decay_t<AccumB>, std::decay_t<Predicate>>(
        std::forward<AccumA>(a), std::forward<AccumB>(b), std::forward<Predicate>(pred)
    );
}

} // namespace accumux
/**
 * @file accumulator_concept.hpp
 * @brief Unified accumulator concepts and interface standards
 * 
 * This file defines the standard interface that all accumux accumulators
 * must implement for seamless composition and interoperability.
 */

#pragma once

#include <type_traits>
#include <concepts>

namespace accumux {

/**
 * @brief Standard accumulator concept
 * 
 * All accumux accumulators must satisfy this concept to be composable.
 * An accumulator is a stateful object that:
 * 1. Processes a stream of values incrementally  
 * 2. Maintains internal state efficiently
 * 3. Can produce a result at any time
 * 4. Can be combined with other accumulators
 */
template<typename T>
concept Accumulator = requires(T acc, typename T::value_type val) {
    // Type requirements
    typename T::value_type;
    
    // Construction requirements
    { T{} } -> std::same_as<T>;                                    // Default constructible
    { T{acc} } -> std::same_as<T>;                                 // Copy constructible
    
    // Core accumulation interface
    { acc += val } -> std::same_as<T&>;                           // Accumulate value
    { acc += acc } -> std::same_as<T&>;                           // Combine accumulators
    
    // Result interface
    { acc.eval() } -> std::convertible_to<typename T::value_type>; // Get result
    { static_cast<typename T::value_type>(acc) };                 // Conversion operator
    
    // Assignment support for composition
    { acc = acc } -> std::same_as<T&>;                            // Copy assignment
};

/**
 * @brief Statistical accumulator concept
 * 
 * Accumulators that compute statistical measures should implement this
 * extended interface for richer composition possibilities.
 */
template<typename T>
concept StatisticalAccumulator = Accumulator<T> && requires(T acc) {
    { acc.size() } -> std::convertible_to<std::size_t>;           // Number of samples
    { acc.mean() } -> std::convertible_to<typename T::value_type>; // Sample mean
};

/**
 * @brief Variance accumulator concept
 * 
 * Accumulators that can compute variance statistics.
 */
template<typename T>
concept VarianceAccumulator = StatisticalAccumulator<T> && requires(T acc) {
    { acc.variance() } -> std::convertible_to<typename T::value_type>;        // Population variance
    { acc.sample_variance() } -> std::convertible_to<typename T::value_type>; // Sample variance
};

/**
 * @brief Accumulator traits for type information
 * 
 * Helper traits to extract information about accumulator types
 * at compile time for template metaprogramming.
 */
template<typename T>
struct accumulator_traits {
    using value_type = typename T::value_type;
    static constexpr bool is_accumulator = Accumulator<T>;
    static constexpr bool is_statistical = StatisticalAccumulator<T>;
    static constexpr bool has_variance = VarianceAccumulator<T>;
};

/**
 * @brief Helper to check if two accumulators are compatible
 * 
 * Two accumulators are compatible if they can work with the same value types.
 */
template<typename T1, typename T2>
constexpr bool compatible_accumulators = 
    Accumulator<T1> && Accumulator<T2> && 
    std::is_same_v<typename T1::value_type, typename T2::value_type>;

} // namespace accumux
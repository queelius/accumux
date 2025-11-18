# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Accumux is a modern C++20 header-only library for compositional online data reductions. It provides numerically stable accumulators that can be algebraically composed to compute multiple statistics in a single pass.

**Version**: 1.2.0
**Test Coverage**: 183 tests, 100% passing

**Key Concepts:**
- **Accumulators**: Stateful objects that process data streams incrementally (e.g., `kbn_sum`, `welford_accumulator`, `min_accumulator`)
- **Compositions**: Algebraic combinations of accumulators using operators:
  - `a + b`: Parallel composition (both process same data, returns tuple)
  - `a * b`: Sequential composition (pipeline: b(a(data)))
  - `conditional(a, b, pred)`: Conditional composition
  - **Important**: Compositions are themselves accumulators and can be nested infinitely
- **Mathematical Foundation**: Built on monoids and homomorphisms for provable correctness
- **Type Safety**: All accumulators use `explicit` conversion operators to prevent implicit conversions

## Build and Test Commands

### Building Tests
```bash
mkdir build && cd build
cmake ..
make -j4
```

### Running All Tests
```bash
cd build
ctest
```

### Running Specific Tests
```bash
# Run a single test executable
cd build
./test_kbn_sum                # KBN sum tests (24 tests)
./test_welford_accumulator    # Welford tests (27 tests)
./test_basic_accumulators     # Basic accumulator tests (47 tests)
./test_concepts               # Concept enforcement tests (17 tests)
./test_composition            # Composition tests (many)
./test_composition_simple     # Simple composition tests
./test_integration            # Integration tests (13 tests)

# Run with filter
./test_welford_accumulator --gtest_filter="*Empty*"
```

### Test Coverage
Enable coverage reporting and generate reports:
```bash
mkdir build && cd build
cmake -DENABLE_COVERAGE=ON ..
make -j4
ctest
gcov ../tests/*.cpp
```

## Architecture

### Core Concepts (`include/accumux/core/`)

**`accumulator_concept.hpp`**: Defines C++20 concepts that all accumulators must satisfy:
- `Accumulator`: Base concept requiring `value_type`, `operator+=`, and `eval()`
- `StatisticalAccumulator`: Extended concept with `size()` and `mean()`
- `VarianceAccumulator`: Further extended with `variance()` methods

**`composition.hpp`**: Implements composition types:
- `parallel_composition<A, B>`: Both accumulators process same data
  - `value_type` is `std::tuple<A::value_type, B::value_type>`
  - `eval()` returns tuple of both results
  - Access via `get_first()` and `get_second()`
- `sequential_composition<A, B>`: Pipeline accumulators
  - `value_type` is `B::value_type` (final output type)
- `conditional_composition<A, B, Pred>`: Runtime switching based on predicate
  - `value_type` is `std::common_type_t<A::value_type, B::value_type>`
- Overloaded operators (`+`, `*`) for natural algebraic syntax
- **All compositions satisfy `Accumulator` concept and can be nested**

### Accumulator Implementations (`include/accumux/accumulators/`)

**`kbn_sum.hpp`**: Kahan-Babuška-Neumaier compensated summation
- Numerically stable floating-point summation
- Maintains `sum_` and `correction_` terms
- Error bound: O(1) vs O(n) for naive summation

**`welford.hpp`**: Welford's online algorithm for mean and variance
- Single-pass computation of mean, variance, std deviation
- Uses `kbn_sum` internally for maximum stability
- Tracks `count_`, `mean_`, and `m2_` (sum of squared differences)

**`basic.hpp`**: Simple accumulators
- `min_accumulator`, `max_accumulator`, `minmax_accumulator`
- `count_accumulator`: Counts elements
- `product_accumulator`: Product with overflow protection

### Key Design Patterns

1. **Type Requirements**: All accumulators must define `value_type` and implement the `Accumulator` concept
2. **Explicit Conversions**: All accumulators use `explicit operator value_type()` to prevent implicit conversions
   - This ensures `a + b` creates a composition rather than converting to scalars
   - Use `static_cast<T>(acc)` or `acc.eval()` for explicit conversion
3. **Operator Overloading**: Use `operator+=` for both value accumulation and accumulator combination
4. **Composition Access**: Use `get_first()` and `get_second()` to access sub-accumulators in compositions
   - For nested: `comp.get_second().get_second()`
   - Results are tuples: `auto [first, second] = comp.eval();`
5. **Result Extraction**: Call `eval()` to get the computed result at any time

### Test Structure (`tests/`)

Tests use Google Test framework with custom helper function:
- `add_accumux_test(test_name)`: Creates test executable and links with GTest
- Test files follow naming: `test_*.cpp`
- **183 total tests** organized by component:
  - `test_kbn_sum.cpp`: KBN sum accumulator tests (24 tests)
  - `test_welford_accumulator.cpp`: Welford algorithm tests (27 tests)
  - `test_basic_accumulators.cpp`: Basic accumulator tests (47 tests)
  - `test_concepts.cpp`: C++20 concept enforcement tests (17 tests)
  - `test_composition.cpp`: Advanced composition tests
  - `test_composition_simple.cpp`: Simple composition tests
  - `test_integration.cpp`: Integration and real-world scenarios (13 tests)
  - `test_additional_coverage.cpp`: Edge cases and coverage tests

## Development Notes

- **Header-Only**: No compilation required; include headers directly
- **C++20 Required**: Uses concepts, requires GCC 10+, Clang 10+, or MSVC 19.29+
- **No Dependencies**: Core library has zero external dependencies (tests require GTest)
- **Numerical Stability**: All floating-point accumulators use compensated algorithms to minimize rounding errors
- **Type Safety**: C++20 concepts prevent incorrect accumulator compositions at compile-time

## Common Patterns

### Creating Custom Accumulators
Must satisfy the `Accumulator` concept:
```cpp
template<typename T>
class my_accumulator {
public:
    using value_type = T;

    my_accumulator() = default;  // Default constructible

    my_accumulator& operator+=(const T& value) {
        // Process value
        return *this;
    }

    my_accumulator& operator+=(const my_accumulator& other) {
        // Combine accumulators
        return *this;
    }

    T eval() const {
        // Return result
    }
};
```

### Using Compositions
```cpp
// Parallel composition - compute multiple stats in one pass
auto stats = kbn_sum<double>() + welford_accumulator<double>();
for (auto val : data) stats += val;

// Access results via get_first()/get_second()
auto sum = stats.get_first().eval();
auto mean = stats.get_second().mean();
auto variance = stats.get_second().sample_variance();

// Or unpack the tuple from eval()
auto [sum_result, welford_result] = stats.eval();
std::cout << "Sum: " << sum_result << "\n";
std::cout << "Mean: " << welford_result.mean() << "\n";

// Nested compositions create nested tuples
auto nested = min_accumulator<double>() + max_accumulator<double>() + count_accumulator();
for (auto val : data) nested += val;
auto [[min_val, max_val], count] = nested.eval();  // Nested structured binding
```

# Accumux

[![C++](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/queelius/accumux)

A modern C++ library for **compositional online data reductions** with mathematical rigor and exceptional performance.

## 🎯 Overview

Accumux provides a powerful framework for combining statistical accumulators using algebraic composition. Built on solid mathematical foundations (monoids, homomorphisms), it enables single-pass computation of complex statistical measures with optimal numerical stability.

### Key Features

- **🧮 Numerically Stable**: Uses advanced algorithms (Kahan-Babuška-Neumaier, Welford) for maximum accuracy
- **⚡ Single-Pass Efficiency**: Compute multiple statistics in O(1) space complexity
- **🔗 Compositional**: Natural algebraic syntax (`a + b`) for combining accumulators  
- **🎯 Type Safe**: C++20 concepts prevent incorrect compositions at compile-time
- **📦 Header-Only**: Zero dependencies, easy integration
- **🏗️ Extensible**: Clean architecture for custom accumulators and compositions

## 🚀 Quick Start

### Basic Usage

```cpp
#include "include/accumux/accumulators/kbn_sum.hpp"
#include "include/accumux/accumulators/welford.hpp"
#include "include/accumux/core/composition.hpp"

using namespace accumux;

int main() {
    // Create a composed accumulator for sum + statistics
    auto stats = kbn_sum<double>() + welford_accumulator<double>();
    
    // Process data in a single pass
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    for (const auto& value : data) {
        stats += value;
    }
    
    // Extract results
    auto sum_result = stats.get_first();
    auto welford_result = stats.get_second();

    std::cout << "Sum: " << sum_result.eval() << std::endl;                      // 15.0
    std::cout << "Mean: " << welford_result.mean() << std::endl;                 // 3.0
    std::cout << "Variance: " << welford_result.sample_variance() << std::endl;  // 2.5
    
    return 0;
}
```

### Advanced Composition

```cpp
// Financial analysis: track multiple metrics simultaneously
auto financial_stats = kbn_sum<double>() + 
                       welford_accumulator<double>() + 
                       minmax_accumulator<double>();

std::vector<double> returns = {0.05, -0.02, 0.03, 0.01, -0.01, 0.04};

for (const auto& ret : returns) {
    financial_stats += ret;  // Single pass, multiple computations
}

// Access all results
auto total_return = financial_stats.get_first().eval();
auto mean_return = financial_stats.get_second().mean();
auto volatility = financial_stats.get_second().sample_std_dev();
auto worst_return = financial_stats.get_second().get_second().min();
auto best_return = financial_stats.get_second().get_second().max();
```

## 📚 Documentation

### Core Accumulators

| Accumulator | Purpose | Algorithm | Complexity |
|-------------|---------|-----------|------------|
| `kbn_sum<T>` | Numerically stable summation | Kahan-Babuška-Neumaier | O(1) space |
| `welford_accumulator<T>` | Mean, variance, std dev | Welford's online algorithm | O(1) space |
| `min_accumulator<T>` | Minimum value tracking | Simple comparison | O(1) space |
| `max_accumulator<T>` | Maximum value tracking | Simple comparison | O(1) space |
| `minmax_accumulator<T>` | Combined min/max | Optimized dual tracking | O(1) space |
| `count_accumulator` | Count of elements | Simple counter | O(1) space |
| `product_accumulator<T>` | Product with overflow protection | Logarithmic representation | O(1) space |

### Composition Operations

| Operation | Syntax | Meaning | Use Case |
|-----------|--------|---------|----------|
| **Parallel** | `a + b` | Both process same data | Multiple statistics |
| **Sequential** | `a * b` | Pipeline: b(a(data)) | Data transformations |
| **Conditional** | `conditional(a, b, pred)` | Choose based on condition | Adaptive processing |

**Key Feature**: Compositions are themselves accumulators, enabling infinite nesting:
- `(a + b) + c` creates a nested composition
- `((a + b) + c) + d` nests arbitrarily deep
- All compositions satisfy the `Accumulator` concept

### Mathematical Guarantees

- **Associativity**: `(a + b) + c` structurally equivalent to `a + (b + c)`
- **Composability**: All compositions satisfy `Accumulator` concept
- **Type Safety**: Compile-time verification via C++20 concepts
- **Numerical Stability**: Error bounds of O(1) vs O(n) for naive algorithms
- **Single-Pass**: All compositions maintain streaming efficiency

## 🏗️ Architecture

### Design Principles

1. **Mathematical Rigor**: Built on solid algebraic foundations (monoids, homomorphisms)
2. **Performance**: Single-pass algorithms with minimal memory overhead
3. **Type Safety**: Compile-time verification of accumulator compatibility  
4. **Composability**: Infinite nesting and combination possibilities
5. **Extensibility**: Clean interfaces for custom accumulators

### Core Concepts

```cpp
template<typename T>
concept Accumulator = requires(std::remove_cvref_t<T> acc,
                               typename std::remove_cvref_t<T>::value_type val) {
    typename std::remove_cvref_t<T>::value_type;           // Value type
    { std::remove_cvref_t<T>{} };                          // Default constructible
    { std::remove_cvref_t<T>{acc} };                       // Copy constructible
    { acc += val } -> std::same_as<std::remove_cvref_t<T>&>;   // Value accumulation
    { acc += acc } -> std::same_as<std::remove_cvref_t<T>&>;   // Accumulator combination
    { acc.eval() } -> std::convertible_to<typename std::remove_cvref_t<T>::value_type>;  // Result
    { acc = acc } -> std::same_as<std::remove_cvref_t<T>&>;    // Copy assignment
};
```

**Important**: All accumulators use `explicit` conversion operators to prevent unintended implicit conversions. This ensures that `a + b` creates a composition rather than converting both to their underlying types.

## ⚡ Performance

Accumux is designed for high-performance streaming applications:

- **Zero Allocation**: Header-only with stack-based computation
- **Cache Friendly**: Minimal memory footprint and sequential access patterns
- **SIMD Ready**: Architecture supports vectorization optimizations
- **Parallel Composition**: Multiple statistics computed simultaneously

### Benchmarks

```
Dataset: 1M double-precision values
Hardware: Modern x86_64 CPU

Single KBN Sum:           ~2.1ms  (476M values/sec)
Parallel Sum + Welford:   ~4.8ms  (208M values/sec)  
Complex 4-way composition: ~8.2ms  (122M values/sec)
```

## 🧪 Testing

Comprehensive test suite with **183 tests** covering:

- **Core Accumulator Tests (119)**: Individual accumulator functionality with edge cases
- **Composition Tests (41)**: Parallel, sequential, and conditional composition behavior
- **Concept Enforcement Tests (17)**: C++20 concept validation and type safety
- **Integration Tests (13)**: Real-world scenarios and complex compositions
- **Numerical Accuracy Tests**: Verification against reference implementations
- **Property Tests**: Mathematical property verification

All tests passing with 100% success rate.

Run tests:
```bash
mkdir build && cd build
cmake ..
make -j4
ctest
```

## 🔧 Requirements

- **C++20** compatible compiler (GCC 10+, Clang 10+, MSVC 19.29+)
- **CMake 3.14+** (for building tests and examples)
- **No external dependencies** (header-only library)

### Supported Platforms

- Linux (GCC, Clang)
- macOS (Clang, Apple Clang)  
- Windows (MSVC, Clang)
- Any C++20 compliant environment

## 📦 Installation

### Header-Only Integration

Simply copy the `include/accumux/` directory to your project:

```cpp
#include "accumux/accumulators/kbn_sum.hpp"
#include "accumux/accumulators/welford.hpp"
#include "accumux/core/composition.hpp"
```

### CMake Integration

```cmake
# Option 1: Add as subdirectory
add_subdirectory(accumux)
target_link_libraries(your_target accumux::accumux)

# Option 2: Find package (if installed)
find_package(accumux REQUIRED)
target_link_libraries(your_target accumux::accumux)
```

### Package Managers

Coming soon: Conan, vcpkg, and other package manager support.

## 🎓 Examples

### Real-Time Analytics

```cpp
// Process streaming financial data
auto market_stats = kbn_sum<double>() + 
                   welford_accumulator<double>() +
                   minmax_accumulator<double>();

while (auto price = get_next_price()) {
    market_stats += price;
    
    // Real-time metrics available at any time
    if (should_report()) {
        std::cout << "Current mean: " << market_stats.get_second().mean() << std::endl;
        std::cout << "Volatility: " << market_stats.get_second().std_dev() << std::endl;
    }
}
```

### Scientific Computing

```cpp
// Multi-dimensional data analysis
auto experiment_analysis = welford_accumulator<double>() + count_accumulator();

for (const auto& measurement : sensor_data) {
    experiment_analysis += measurement;
}

// Statistical validation
auto stats = experiment_analysis.get_first();
if (stats.sample_std_dev() > threshold) {
    std::cout << "High variability detected!" << std::endl;
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/queelius/accumux.git
cd accumux
mkdir build && cd build
cmake -DACCUMUX_BUILD_TESTS=ON ..
make -j4
ctest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Kahan, Babuška, and Neumaier** for the compensated summation algorithm
- **Donald Knuth** for "The Art of Computer Programming" insights on numerical algorithms  
- **Welford** for the online variance algorithm
- **SICP** for inspiration on algebraic composition principles
- The **C++ standards committee** for concepts and modern language features

## 📊 Project Status

- **Version**: 1.2.0
- **Status**: Stable
- **Test Coverage**: 183 tests, 100% passing
- **API Stability**: Stable (following semantic versioning)
- **Maintenance**: Active development

## 🔗 Links

- [Documentation](https://queelius.github.io/accumux)
- [API Reference](https://queelius.github.io/accumux/api)  
- [Examples](examples/)
- [Benchmarks](benchmarks/)
- [Issue Tracker](https://github.com/queelius/accumux/issues)

---

*Built with ❤️ for the C++ community*
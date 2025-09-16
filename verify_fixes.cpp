#include "include/accumux/accumulators/kbn_sum.hpp"
#include "include/accumux/accumulators/welford.hpp"
#include <iostream>
#include <vector>

int main() {
    using namespace accumux;
    
    std::cout << "Testing accumux library fixes..." << std::endl;
    
    // Test KBN sum
    kbn_sum<double> sum;
    sum += 1.0;
    sum += 2.0;
    sum += 3.0;
    
    std::cout << "KBN Sum: " << sum.eval() << " (expected: 6.0)" << std::endl;
    
    // Test Welford accumulator
    welford_accumulator<double> welford;
    welford += 1.0;
    welford += 2.0;
    welford += 3.0;
    
    std::cout << "Welford Mean: " << welford.mean() << " (expected: 2.0)" << std::endl;
    std::cout << "Welford Size: " << welford.size() << " (expected: 3)" << std::endl;
    std::cout << "Welford Variance: " << welford.variance() << " (expected: ~0.667)" << std::endl;
    
    // Test factory functions
    auto kbn_acc = make_kbn_sum<double>(10.0);
    auto welford_acc = make_welford_accumulator<double>();
    
    std::cout << "Factory KBN: " << kbn_acc.eval() << " (expected: 10.0)" << std::endl;
    
    // Test combining accumulators
    kbn_sum<double> sum1(5.0);
    kbn_sum<double> sum2(7.0);
    auto result = sum1 + sum2;
    
    std::cout << "Combined Sum: " << result.eval() << " (expected: 12.0)" << std::endl;
    
    std::cout << "All tests completed successfully!" << std::endl;
    return 0;
}
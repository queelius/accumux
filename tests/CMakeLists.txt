# Test configuration
include(GoogleTest)

# Helper function to create tests
function(add_accumux_test test_name)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name} GTest::gtest GTest::gtest_main)
    target_include_directories(${test_name} PRIVATE ${CMAKE_SOURCE_DIR})
    gtest_discover_tests(${test_name})
endfunction()

# Add test files
add_accumux_test(test_kbn_sum)
add_accumux_test(test_welford_accumulator)
add_accumux_test(test_utility_coverage)
add_accumux_test(test_basic_accumulators)
add_accumux_test(test_integration)
add_accumux_test(test_concepts)
add_accumux_test(test_composition)
add_accumux_test(test_composition_simple)
add_accumux_test(test_expressions_basic)
add_accumux_test(test_additional_coverage)

# New v2.0 tests
add_accumux_test(test_extended_accumulators)
add_accumux_test(test_variadic_composition)
add_accumux_test(test_distributed)
add_accumux_test(test_algebra)
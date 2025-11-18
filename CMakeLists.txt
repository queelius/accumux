cmake_minimum_required(VERSION 3.20)

project(accumux
    VERSION 1.2.0
    DESCRIPTION "Modern C++ library for compositional online data reductions"
    LANGUAGES CXX
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Coverage options
option(ENABLE_COVERAGE "Enable coverage reporting" OFF)
if(ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(--coverage -fprofile-arcs -ftest-coverage)
        add_link_options(--coverage)
    endif()
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Enable testing
enable_testing()

# Find required packages
find_package(GTest QUIET)

if(GTest_FOUND)
    include(GoogleTest)
    add_subdirectory(tests)
else()
    message(STATUS "Google Test not found. Tests will not be built.")
endif()
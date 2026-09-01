cmake_minimum_required(VERSION 3.24)

project(NSFEM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP REQUIRED)

include(FetchContent)
include(GoogleTest)

FetchContent_Declare(
	googletest
	URL https://github.com/google/googletest/archive/d94a7326e97c9d564950e36c48ac402ba4933985.zip
)

FetchContent_MakeAvailable(googletest)

enable_testing()

add_library(fem
	src/mesh/parser.cpp
	src/geometry/triangle_geometry.cpp
	src/assemble/assemble.cpp
	src/basis/lagrange_p1_triangle.cpp
	src/quadrature/quadrature.cpp
	src/linalg/matrix.cpp
	src/linalg/conjugate_gradient.cpp
	src/convergence/convergence.cpp
	src/helpers/helpers.cpp
)

target_include_directories(fem PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(fem PRIVATE OpenMP::OpenMP_CXX)

target_compile_options(fem PUBLIC -Wall -Wextra)

add_executable(main src/main.cpp)

target_link_libraries(main PRIVATE fem)

add_executable(tests tests/hello_test.cpp)

target_link_libraries(tests PRIVATE fem GTest::gtest_main)

gtest_discover_tests(tests)


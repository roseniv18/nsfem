cmake_minimum_required(VERSION 3.24)

project(NSFEM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

add_executable(main
    main.cpp
	mesh/parser.cpp
	geometry/affine.cpp
	assemble/assemble.cpp
	basis/lagrange_p1_triangle.cpp
	quadrature/quadrature.cpp
)

target_compile_options(main PUBLIC -Wall -Wextra)

target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

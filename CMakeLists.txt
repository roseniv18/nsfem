cmake_minimum_required(VERSION 3.24)

project(NSFEM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

add_executable(main
    main.cpp
	mesh/parser.cpp
	geometry/affine.cpp
)

target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

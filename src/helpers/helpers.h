#ifndef HELPERS_H
#define HELPERS_H

#include <string>
#include "mesh/parser.h"

std::string strip_quotes(std::string& str);

// n - resolution of mesh
Mesh make_unit_square_mesh(int n);

// tolerance (used for tests that use EXPECT_NEAR)
const double test_dtol = 1e-12;

#endif

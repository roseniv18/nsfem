#ifndef HELPERS_H
#define HELPERS_H

#include <string>
#include "mesh/parser.h"

std::string strip_quotes(std::string& str);

// n - resolution of mesh
Mesh make_unit_square_mesh(int n);

#endif

#ifndef FEM_MATRIX_H
#define FEM_MATRIX_H

#include <array>
#include "mesh/parser.h"

typedef std::array<std::array<double, 3>, 3> local_matr;

// generate a local stiffness matrix for given element
local_matr generate_ls_matrix(const Mesh& mesh, const Element& element);

// assemble global stiffness matrix
std::vector<std::vector<double>> assemble_gs_matrix(const Mesh& mesh);

#endif

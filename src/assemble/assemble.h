#ifndef FEM_MATRIX_H
#define FEM_MATRIX_H

#include <array>
#include "basis/lagrange_p1_triangle.h"
#include "linalg/matrix.h"
#include "mesh/parser.h"

typedef std::array<double, 3> local_vec;

// generate a local stiffness matrix for given element
Matrix<double> generate_ls_matrix(const Element& element, const Mesh& mesh);

// assemble global stiffness matrix
Matrix<double> assemble_gs_matrix(const Mesh& mesh);

// generate a local load vector
local_vec generate_loc_vector(const Element& element,
                              double (*func)(const Point2D&),
                              const Mesh& mesh);

// assemble global load vector
std::vector<double> assemble_gl_vector(const Mesh& mesh,
                                       double (*f)(const Point2D&));

// apply Dirichlet boundary conditions
void apply_dirichlet_bc(Matrix<double>& K,
                        std::vector<double>& f,
                        const std::unordered_map<int, double>& dirichlet_nodes);

#endif

#ifndef FEM_MATRIX_H
#define FEM_MATRIX_H

#include <array>
#include "basis/lagrange_p1_triangle.h"
#include "mesh/parser.h"

typedef std::array<std::array<double, 3>, 3> local_matr;
typedef std::array<double, 3> local_vec;

// generate a local stiffness matrix for given element
local_matr generate_ls_matrix(const Element& element);

// assemble global stiffness matrix
std::vector<std::vector<double>> assemble_gs_matrix(const Mesh& mesh);

// generate a local load vector
local_vec generate_loc_vector(const Element& element,
                              double (*func)(const Point2D&));

// assemble global load vector
std::vector<double> assemble_gl_vector(const Mesh& mesh,
                                       double (*f)(const Point2D&));

// apply Dirichlet boundary conditions
void apply_dirichlet_bc(std::vector<std::vector<double>>& K,
                        std::vector<double>& f,
                        const std::unordered_map<int, double>& dirichlet_nodes);

#endif

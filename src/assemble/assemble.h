#ifndef FEM_MATRIX_H
#define FEM_MATRIX_H

#include <stdlib.h>
#include <array>
#include "assemble.h"
#include "basis/lagrange_p1_triangle.h"
#include "geometry/triangle_geometry.h"
#include "linalg/matrix.h"
#include "math/function.h"
#include "mesh/mesh.h"

typedef std::array<double, 3> local_vec;

// generate a local stiffness matrix for given element
Matrix<double> gen_local_stiffness_matr(const TriangleGEO& element);

// assemble global stiffness matrix
Matrix<double> asm_global_stiffness_matr(const Mesh& mesh);

// generate a local mass matrix for given element
Matrix<double> gen_local_mass_matr(const TriangleGEO& element);

// assemble global mass matrix
Matrix<double> asm_global_mass_matr(const Mesh& mesh);

// generate a local load vector
local_vec gen_local_vec(const TriangleGEO& element, STFunction func, double t);

// assemble global load vector
std::vector<double> asm_global_vec(const Mesh& mesh, STFunction func, double t);

// apply Dirichlet boundary conditions
void apply_dirichlet_bc(Matrix<double>& A,
                        std::vector<double>& vec,
                        const std::vector<bool>& is_dirichlet,
                        const std::vector<double>& dirichlet_vals);

void apply_dirichlet_bc_matr(Matrix<double>& A,
                             const std::vector<bool>& is_dirichlet);

void apply_dirichlet_bc_vec(Matrix<double>& A,
                            std::vector<double>& vec,
                            const std::vector<double>& dirichlet_vals,
                            const std::vector<bool>& is_dirichlet);

#endif

#ifndef FEM_MATRIX_H
#define FEM_MATRIX_H

#include <stdlib.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <array>
#include "assemble.h"
#include "basis/lagrange_p1_triangle.h"
#include "geometry/triangle_geometry.h"
#include "math/function.h"
#include "mesh/mesh.h"

using Eigen::MatrixXd, Eigen::VectorXd;
using SparseMatrix = Eigen::SparseMatrix<double>;

// generate a local stiffness matrix for given element
MatrixXd gen_local_stiffness_matr(const TriangleGEO& element);

// assemble global stiffness matrix
SparseMatrix asm_global_stiffness_matr(const Mesh& mesh);

// generate a local mass matrix for given element
MatrixXd gen_local_mass_matr(const TriangleGEO& element);

// assemble global mass matrix
SparseMatrix asm_global_mass_matr(const Mesh& mesh);

// generate a local load vector
VectorXd gen_local_vec(const TriangleGEO& element, STFunction func, double t);

// assemble global load vector
VectorXd asm_global_vec(const Mesh& mesh, STFunction func, double t);

// apply Dirichlet boundary conditions
void apply_dirichlet_bc(SparseMatrix& A,
                        VectorXd& vec,
                        const std::vector<bool>& is_dirichlet,
                        const std::vector<double>& dirichlet_vals);

void apply_dirichlet_bc_matr(SparseMatrix& A,
                             const std::vector<bool>& is_dirichlet);

void apply_dirichlet_bc_vec(const SparseMatrix& A,
                            VectorXd& vec,
                            const std::vector<bool>& is_dirichlet,
                            const std::vector<double>& dirichlet_vals);

#endif

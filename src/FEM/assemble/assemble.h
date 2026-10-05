#ifndef FEM_MATRIX_H
#define FEM_MATRIX_H

#include <stdlib.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <array>
#include <memory>
#include "FEM/finite_element/dof_handler.h"
#include "FEM/finite_element/finite_element.h"
#include "FEM/mapping/fe_mapping.h"
#include "math/function.h"
#include "mesh/mesh.h"

using Eigen::MatrixXd, Eigen::VectorXd;
using SparseMatrix = Eigen::SparseMatrix<double>;

// generate a local stiffness matrix for given element
MatrixXd gen_local_stiffness_matr(const FEMap2D& mapping, const FE& fe);

// assemble global stiffness matrix
SparseMatrix asm_global_stiffness_matr(const Mesh& mesh,
                                       const FE& fe,
                                       const DOFHandler& dofh);

// generate a local mass matrix for given element
MatrixXd gen_local_mass_matr(const FEMap2D& mapping, const FE& fe);

// assemble global mass matrix
SparseMatrix asm_global_mass_matr(const Mesh& mesh,
                                  const FE& fe,
                                  const DOFHandler& dofh);

// generate a local load vector
VectorXd gen_local_vec(const FEMap2D& mapping,
                       STFunction func,
                       const FE& element,
                       double t);

// assemble global load vector
VectorXd asm_global_vec(const Mesh& mesh,
                        const FE& fe,
                        const DOFHandler& dofh,
                        STFunction func,
                        double t);

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

#ifndef FEM_ASSEMBLE_H
#define FEM_ASSEMBLE_H

#include <stdlib.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <array>
#include <memory>
#include "FEM/finite_element/dof_handler.h"
#include "FEM/finite_element/finite_element.h"
#include "FEM/mapping/fe_mapping.h"
#include "FEM/mapping/phys_node.h"
#include "FEM/mapping/ref_node.h"
#include "FEM/quadrature/quadrature.h"
#include "math/function.h"
#include "mesh/mesh.h"

namespace nsfem::assembly {

using Eigen::MatrixXd, Eigen::VectorXd;
using SparseMatrix = Eigen::SparseMatrix<double>;
using nsfem::fem::FE, nsfem::fem::DOFHandler;
using nsfem::mapping::FEMap2D;
using nsfem::mesh::Mesh, nsfem::mesh::Element, nsfem::mesh::is_triangle_type;

// generate a local stiffness matrix for given element
MatrixXd gen_local_stiffness_matr(const FEMap2D& mapping, const FE& fe);

// assemble global stiffness matrix
SparseMatrix asm_stiffness_matrix(const Mesh& mesh,
                                  const FE& fe,
                                  const DOFHandler& dofh);

// generate a local mass matrix for given element
MatrixXd gen_local_mass_matr(const FEMap2D& mapping, const FE& fe);

// assemble global mass matrix
SparseMatrix asm_mass_matrix(const Mesh& mesh,
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

// generate local velocity stiffness matrix (stokes)
MatrixXd gen_local_velocity_stiffness_matr(const FEMap2D& mapping,
                                           const FE& fe,
                                           const double viscosity);

// generate local velocity-pressure coupling matrix (stokes)
MatrixXd gen_local_velocity_pressure_matr(const FEMap2D& mapping,
                                          const FE& velocity_fe,
                                          const FE& pressure_fe);

// generate local forcing vector for Stokes
VectorXd gen_local_stokes_forcing_vector(const FEMap2D& mapping,
                                         const FE& velocity_fe,
                                         const VecFunction& forcing);

// assemble Stokes system
SparseMatrix asm_stokes_system(const Mesh& mesh,
                               const FE& velocity_fe,
                               const FE& pressure_fe,
                               const DOFHandler& velocity_dofh,
                               const DOFHandler& pressure_dofh,
                               VecFunction stokes_f_func);

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

}  // namespace nsfem::assembly

#endif

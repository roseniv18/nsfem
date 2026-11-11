#include "stokes.h"

namespace nsfem::solvers {
Stokes::Stokes(const Mesh& mesh_,
               const FE& velocity_fe_,
               const FE& pressure_fe_,
               const DOFHandler& velocity_dofh_,
               const DOFHandler& pressure_dofh_,
               StokesFunction stokes_f_func_,
               VecFunction stokes_dir_func_,
               double viscosity_)
    : mesh(mesh_),
      velocity_fe(velocity_fe_),
      pressure_fe(pressure_fe_),
      velocity_dofh(velocity_dofh_),
      pressure_dofh(pressure_dofh_),
      global_ndofs(2 * velocity_dofh_.get_global_ndofs() +
                   pressure_dofh_.get_global_ndofs()),
      stokes_f_func(stokes_f_func_),
      stokes_dir_func(stokes_dir_func_),
      viscosity(viscosity_) {
  rhs_vec.resize(global_ndofs);
}

Eigen::VectorXd Stokes::solve() {
  // Assemble velocity stiffness matrix
  // Assmeble velocity-pressure coupling matrix
  Eigen::SparseMatrix G = asm_stokes_matrix(
      mesh, velocity_fe, pressure_fe, velocity_dofh, pressure_dofh, viscosity);
  // Assemble RHS
  rhs_vec = asm_stokes_rhs(mesh, velocity_fe, velocity_dofh, pressure_dofh,
                           stokes_f_func, viscosity);

  const Eigen::Index n_velocity_dofs =
      static_cast<Eigen::Index>(velocity_dofh.get_global_ndofs());

  const Eigen::Index n_pressure_dofs =
      static_cast<Eigen::Index>(pressure_dofh.get_global_ndofs());

  const Eigen::Index pressure_offset = n_velocity_dofs * 2;

  const Eigen::Index total_dofs = 2 * n_velocity_dofs + n_pressure_dofs;

  if (G.rows() != total_dofs || G.cols() != total_dofs ||
      rhs_vec.size() != total_dofs) {
    throw std::runtime_error(
        "Stokes: Assembled system dimensions are incorrect!");
  }

  /**
   * Apply Dirichlet BCs
   */
  std::vector<bool> is_dirichlet(total_dofs, false);
  std::vector<double> dirichlet_vals(total_dofs, 0.0);

  const std::vector<bool> velocity_dirichlet_dofs =
      nsfem::mesh::get_dirichlet_dofs(mesh, velocity_dofh);

  /**
   * Account for both velocity components
   */
  for (Eigen::Index i = 0; i < n_velocity_dofs; i++) {
    if (!velocity_dirichlet_dofs.at(i))
      continue;

    is_dirichlet.at(i) = true;
    is_dirichlet.at(i + n_velocity_dofs) = true;
  }

  /**
   * Pin an arbitrary pressure DOF (set it to dirichlet val)
   * This is done to remove the singularity from the global matrix G, which
   * arises from the pressure being defined only up to a constant. Another
   * approach would be  to use a Lagrange multiplier.
   */
  constexpr Eigen::Index pinned_pressure_dof = 0;

  const Eigen::Index global_pressure_dof =
      pinned_pressure_dof + pressure_offset;

  // the process of pinning the pressure DOF is equivalent to applying Dirichlet
  is_dirichlet.at(global_pressure_dof) = true;

  nsfem::assembly::apply_dirichlet_bc(G, rhs_vec, is_dirichlet, dirichlet_vals);

  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;

  solver.compute(G);

  if (solver.info() != Eigen::Success) {
    throw std::runtime_error("Stokes: Sparse LU factorization failed!");
  }

  Eigen::VectorXd solution = solver.solve(rhs_vec);

  if (solver.info() != Eigen::Success || !solution.allFinite()) {
    throw std::runtime_error("Stokes: Linear system solve failed!");
  }

  return solution;
}
}  // namespace nsfem::solvers

#include "stokes.h"

namespace nsfem::solvers {

Stokes::Stokes(const Mesh& mesh_,
               const FE& velocity_fe_,
               const FE& pressure_fe_,
               const DOFHandler& velocity_dofh_,
               const DOFHandler& pressure_dofh_,
               STFunction stokes_f_func_,
               STFunction stokes_dir_func_)
    : mesh(mesh_),
      velocity_fe(velocity_fe_),
      pressure_fe(pressure_fe_),
      velocity_dofh(velocity_dofh_),
      pressure_dofh(pressure_dofh_),
      global_ndofs(2 * velocity_dofh_.get_global_ndofs() +
                   pressure_dofh_.get_global_ndofs()),
      stokes_f_func(stokes_f_func_),
      stokes_dir_func(stokes_dir_func_) {
  rhs_vec.resize(global_ndofs);
}

VectorXd Stokes::solve() {
  // Assemble velocity stiffness matrix
  // Assmeble velocity-pressure coupling matrix
  // Assemble RHS
  // Apply discrete zero-mean pressure condition
  // Apply Dirichlet BC
  // Solve linear system Eigen
}
}  // namespace nsfem::solvers

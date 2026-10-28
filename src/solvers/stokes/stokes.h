#ifndef STOKES_H
#define STOKES_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include "FEM/assemble/assemble.h"
#include "FEM/finite_element/dof_handler.h"
#include "FEM/finite_element/finite_element.h"
#include "math/function.h"
#include "mesh/mesh.h"

namespace nsfem::solvers {
using Eigen::VectorXd;
using nsfem::fem::FE, nsfem::fem::DOFHandler;
using nsfem::mesh::Mesh;

class Stokes {
 public:
  Stokes(const Mesh& mesh,
         const FE& velocity_fe,
         const FE& pressure_fe,
         const DOFHandler& velocity_dofh,
         const DOFHandler& pressure_dofh,
         STFunction stokes_f_func,
         STFunction stokes_dir_func);

  VectorXd solve();

 private:
  const Mesh& mesh;
  const FE& velocity_fe;
  const FE& pressure_fe;
  const DOFHandler& velocity_dofh;
  const DOFHandler& pressure_dofh;
  const std::size_t global_ndofs;
  STFunction stokes_f_func;
  STFunction stokes_dir_func;

  VectorXd rhs_vec;
};
}  // namespace nsfem::solvers

#endif

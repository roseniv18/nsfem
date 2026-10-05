#ifndef POISSON_H
#define POISSON_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include "FEM/assemble/assemble.h"
#include "FEM/finite_element/dof_handler.h"
#include "FEM/finite_element/finite_element.h"
#include "linalg/conjugate_gradient.h"
#include "math/function.h"
#include "mesh/mesh.h"

using Eigen::VectorXd, Eigen::MatrixXd;
using SparseMatrix = Eigen::SparseMatrix<double>;

class Poisson {
 public:
  Poisson(const Mesh& mesh,
          const FE& fe,
          const DOFHandler& dofh,
          STFunction f_func,
          STFunction dir_func);
  VectorXd solve();

 private:
  const Mesh& mesh;
  const FE& fe;
  const DOFHandler& dofh;
  STFunction f_func;
  STFunction dir_func;

  VectorXd rhs_vec;
};

#endif

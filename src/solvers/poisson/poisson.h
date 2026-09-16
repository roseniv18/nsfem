#ifndef POISSON_H
#define POISSON_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include "assemble/assemble.h"
#include "linalg/conjugate_gradient.h"
#include "math/function.h"
#include "mesh/mesh.h"

using Eigen::VectorXd, Eigen::MatrixXd;
using SparseMatrix = Eigen::SparseMatrix<double>;

class Poisson {
 public:
  Poisson(const Mesh& mesh, STFunction f_func, STFunction dir_func);
  VectorXd solve(const Mesh& mesh);

 private:
  STFunction f_func;
  STFunction dir_func;

  VectorXd rhs_vec;
};

#endif

#ifndef HEAT_EQ_H
#define HEAT_EQ_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <vector>
#include "FEM/assemble/assemble.h"
#include "FEM/convergence/convergence.h"
#include "FEM/finite_element/finite_element.h"
#include "linalg/conjugate_gradient.h"
#include "mesh/mesh.h"

using Eigen::MatrixXd, Eigen::VectorXd;
using SparseMatrix = Eigen::SparseMatrix<double>;
using std::exp;
using std::sin;
using std::numbers::pi;

/** Solve the 2D heat equation on the unit square
 * du/dt​ − Δu = 0
 * u(x,y,0) = sin(πx) * sin(πy)
 * u(.,.,t) = 0, dOmega (homogeneous Dirichlet BC)
 */

#include "FEM/finite_element/finite_element.h"

class HeatEq {
 public:
  HeatEq(const Mesh& mesh,
         const FE& fe,
         const DOFHandler& dofh,
         const double dt,
         double T,
         STFunction h_func,
         STFunction h_dir_func);
  void update_rhs(double t);
  VectorXd solve();

 private:
  const Mesh& mesh;
  const FE& fe;
  const DOFHandler& dofh;

  VectorXd initial_condition;
  VectorXd rhs_vec;

  STFunction h_func;
  STFunction h_dir_func;

  double dt{};
  double T{};
  int n_steps{};
};

#endif

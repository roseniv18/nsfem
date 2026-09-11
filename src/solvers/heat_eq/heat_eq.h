#ifndef HEAT_EQ_H
#define HEAT_EQ_H

#include <cmath>
#include <vector>
#include "assemble/assemble.h"
#include "convergence/convergence.h"
#include "linalg/conjugate_gradient.h"
#include "mesh/mesh.h"

using std::exp;
using std::sin;
using std::numbers::pi;

/** Solve the 2D heat equation on the unit square
 * du/dt​ − Δu = 0
 * u(x,y,0) = sin(πx) * sin(πy)
 * u(.,.,t) = 0, dOmega (homogeneous Dirichlet BC)
 */

class HeatEq {
 public:
  HeatEq(const Mesh& mesh,
         const double dt,
         double T,
         STFunction h_func,
         STFunction h_dir_func);
  void update_rhs(const Mesh& mesh, double t);
  std::vector<double> solve(const Mesh& mesh);

 private:
  std::vector<double> initial_condition;
  std::vector<double> rhs_vec;

  STFunction h_func;
  STFunction h_dir_func;

  double dt{};
  double T{};
  int n_steps{};
};

#endif

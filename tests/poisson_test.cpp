#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <iostream>
#include <numbers>
#include <vector>

#include "FEM/assemble/assemble.h"
#include "FEM/convergence/convergence.h"
#include "FEM/finite_element/dof_handler.h"
#include "FEM/finite_element/finite_element.h"
#include "helpers/helpers.h"
#include "linalg/conjugate_gradient.h"
#include "mesh/parser.h"
#include "solvers/poisson/poisson.h"

using Eigen::VectorXd;
using SparseMatrix = Eigen::SparseMatrix<double>;

TEST(PoissonTest, L2Convergence) {
  const std::vector<int> resolutions{4, 8, 16, 32};

  std::vector<double> errors;

  for (const int n : resolutions) {
    Mesh mesh = make_unit_square_mesh(n);

    const std::unique_ptr<FE> fe = FE::build_fe_type(FEType::P1);
    const DOFHandler dofh(mesh, *fe);

    Poisson poisson(mesh, *fe, dofh, h_func, h_dir_func);

    const VectorXd solution = poisson.solve();

    const double error =
        global_l2_err(mesh, *fe, dofh, solution, h_sol_func, 0.05);

    EXPECT_LT(error, 0.1);
  }

  // ------------------------------------------------------------
  // Check convergence rate
  // ------------------------------------------------------------

  for (std::size_t i = 1; i < errors.size(); ++i) {
    const double rate = std::log(errors[i - 1] / errors[i]) / std::log(2.0);

    EXPECT_GT(rate, 1.8);
    EXPECT_LT(rate, 2.2);
  }
}

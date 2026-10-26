#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <iostream>
#include <numbers>
#include <vector>

#include "FEM/assemble/assemble.h"
#include "FEM/convergence/convergence.h"
#include "FEM/finite_element/P1_element.h"
#include "FEM/finite_element/P2_element.h"
#include "FEM/finite_element/dof_handler.h"
#include "FEM/finite_element/finite_element.h"
#include "helpers/helpers.h"
#include "linalg/conjugate_gradient.h"
#include "mesh/parser.h"
#include "solvers/poisson/poisson.h"

using Eigen::VectorXd;
using SparseMatrix = Eigen::SparseMatrix<double>;
using namespace nsfem::convergence;
using nsfem::fem::P1_FE, nsfem::fem::P2_FE;
using nsfem::solvers::Poisson;

/**
 * Poisson 2D with P1 elements
 */
TEST(PoissonTest, L2ConvergenceP1) {
  const std::vector<int> resolutions{4, 8, 16, 32};

  std::vector<double> errors;
  errors.reserve(resolutions.size());

  for (const int n : resolutions) {
    Mesh mesh = make_unit_square_mesh(n);

    const P1_FE fe;
    const DOFHandler dofh(mesh, fe);

    Poisson poisson(mesh, fe, dofh, func, dir_func);

    const VectorXd solution = poisson.solve();

    ASSERT_EQ(solution.size(), dofh.get_global_ndofs());
    ASSERT_TRUE(solution.allFinite());

    const double error = global_l2_err(mesh, fe, dofh, solution, sol_func, 0.0);

    ASSERT_TRUE(std::isfinite(error));
    ASSERT_GT(error, 0.0);

    errors.push_back(error);
  }

  // ------------------------------------------------------------
  // Check convergence rate
  // ------------------------------------------------------------

  for (std::size_t i = 1; i < errors.size(); ++i) {
    // expect the error to decrease, when refining mesh
    EXPECT_LT(errors[i], errors[i - 1]);

    const double rate = std::log(errors[i - 1] / errors[i]) / std::log(2.0);

    std::cout << "rate[" << i - 1 << " -> " << i << "] = " << rate << '\n';

    EXPECT_GT(rate, 1.8);
    EXPECT_LT(rate, 2.2);
  }
}

/**
 * Poisson 2D with P2 elements
 */
TEST(PoissonTest, L2ConvergenceP2) {
  const std::vector<int> resolutions{4, 8, 16, 32};

  std::vector<double> errors;
  errors.reserve(resolutions.size());

  for (const int n : resolutions) {
    Mesh mesh = make_unit_square_mesh(n);

    const P2_FE fe;
    const DOFHandler dofh(mesh, fe);

    Poisson poisson(mesh, fe, dofh, func, dir_func);

    const VectorXd solution = poisson.solve();

    ASSERT_EQ(solution.size(), dofh.get_global_ndofs());
    ASSERT_TRUE(solution.allFinite());

    const double error = global_l2_err(mesh, fe, dofh, solution, sol_func, 0.0);

    ASSERT_TRUE(std::isfinite(error));
    ASSERT_GT(error, 0.0);

    errors.push_back(error);
  }

  // ------------------------------------------------------------
  // Check convergence rate
  // ------------------------------------------------------------

  for (std::size_t i = 1; i < errors.size(); ++i) {
    // expect the error to decrease, when refining mesh
    EXPECT_LT(errors[i], errors[i - 1]);

    const double rate = std::log(errors[i - 1] / errors[i]) / std::log(2.0);

    std::cout << "rate[" << i - 1 << " -> " << i << "] = " << rate << '\n';

    EXPECT_GT(rate, 2.5);
    EXPECT_LT(rate, 3.5);
  }
}

/**
 * Test if the Poisson solver imposes Dirichlet BCs correctly
 */
TEST(PoissonTest, CorrectDirichletBC) {
  Mesh mesh = make_unit_square_mesh(8);

  const P1_FE fe;
  const DOFHandler dofh(mesh, fe);

  Poisson poisson(mesh, fe, dofh, func, dir_func);
  const VectorXd solution = poisson.solve();

  ASSERT_TRUE(solution.allFinite());

  const auto is_dirichlet = get_dirichlet_dofs(mesh, dofh);
  const auto dirichlet_values =
      get_dirichlet_values(mesh, fe, dofh, is_dirichlet, dir_func, 0.0);

  for (Eigen::Index i = 0; i < solution.size(); i++) {
    if (is_dirichlet[i]) {
      EXPECT_NEAR(solution[i], dirichlet_values[i], test_dtol);
    }
  }
}

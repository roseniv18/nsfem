#include "solvers/heat_eq/heat_eq.h"
#include <gtest/gtest.h>
#include <memory>
#include "FEM/convergence/convergence.h"
#include "FEM/finite_element/P1_element.h"
#include "FEM/finite_element/P2_element.h"
#include "FEM/finite_element/dof_handler.h"
#include "FEM/finite_element/finite_element.h"
#include "helpers/helpers.h"
#include "mesh/mesh.h"

// Check that the FEM heat equation solution matches the analytical solution
TEST(HeatEqTest, MatchesAnalyticalSolution) {
  const double dt = 0.01;
  const double T = 0.5;

  Mesh mesh = make_unit_square_mesh(16);

  const P1_FE fe;
  const DOFHandler dofh(mesh, fe);

  HeatEq heat_eq(mesh, fe, dofh, dt, T, h_func, h_dir_func);

  const VectorXd solution = heat_eq.solve();

  ASSERT_TRUE(solution.allFinite());

  const double error = global_l2_err(mesh, fe, dofh, solution, h_sol_func, T);

  EXPECT_TRUE(std::isfinite(error));

  EXPECT_LT(error, 0.05);
}

/**
 * For a sufficiently smooth solution, implicit Euler is O(τ)
 */
TEST(HeatEqTest, FirstOrderTemporalConvergence) {
  Mesh mesh = make_unit_square_mesh(64);

  const P1_FE fe;
  const DOFHandler dofh(mesh, fe);

  const double T = 0.1;
  const std::vector<double> timesteps{0.02, 0.01, 0.005};

  std::vector<double> errors;
  errors.reserve(timesteps.size());

  for (const double dt : timesteps) {
    HeatEq heat_eq(mesh, fe, dofh, dt, T, h_func, h_dir_func);

    const VectorXd solution = heat_eq.solve();

    ASSERT_TRUE(solution.allFinite());

    const double error = global_l2_err(mesh, fe, dofh, solution, h_sol_func, T);

    ASSERT_TRUE(std::isfinite(error));
    ASSERT_GT(error, 0.0);

    errors.push_back(error);
  }

  for (std::size_t i = 1; i < errors.size(); i++) {
    EXPECT_LT(errors[i], errors[i - 1]);

    const double rate = std::log(errors[i - 1] / errors[i]) / std::log(2.0);

    EXPECT_GT(rate, 0.8);
    EXPECT_LT(rate, 1.2);
  }
}

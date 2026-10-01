#include "solvers/heat_eq/heat_eq.h"
#include <gtest/gtest.h>
#include "FEM/convergence/convergence.h"
#include "helpers/helpers.h"
#include "mesh/mesh.h"

// Check that the FEM heat equation solution matches the analytical solution
TEST(HeatEqTest, MatchesAnalyticalSolution) {
  Mesh mesh = make_unit_square_mesh(16);

  HeatEq heat_eq(mesh, 0.01, 0.05, h_func, h_dir_func);

  const VectorXd solution = heat_eq.solve(mesh);

  const double error = global_l2_err(mesh, solution, h_sol_func, 0.05);

  EXPECT_LT(error, 0.5);
}

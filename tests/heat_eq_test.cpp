#include "solvers/heat_eq/heat_eq.h"
#include <gtest/gtest.h>
#include <memory>
#include "FEM/convergence/convergence.h"
#include "FEM/finite_element/dof_handler.h"
#include "FEM/finite_element/finite_element.h"
#include "helpers/helpers.h"
#include "mesh/mesh.h"

// Check that the FEM heat equation solution matches the analytical solution
TEST(HeatEqTest, MatchesAnalyticalSolution) {
  Mesh mesh = make_unit_square_mesh(16);

  const std::unique_ptr<FE> fe = FE::build_fe_type(FEType::P1);
  const DOFHandler dofh(mesh, *fe);

  HeatEq heat_eq(mesh, *fe, dofh, 0.01, 0.05, h_func, h_dir_func);

  const VectorXd solution = heat_eq.solve();

  const double error =
      global_l2_err(mesh, *fe, dofh, solution, h_sol_func, 0.05);

  EXPECT_LT(error, 0.5);
}

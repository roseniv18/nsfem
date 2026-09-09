#include "heat_eq/heat_eq.h"
#include <gtest/gtest.h>
#include "assemble/assemble.h"
#include "heat_eq/implicit_euler.h"
#include "helpers/helpers.h"

TEST(HeatEqTest, ImplicitEulerStep) {
  Mesh mesh = make_unit_square_mesh(64);

  HeatEq heat_eq(mesh, 0.01, 0.01);

  // One step in time using implicit Euler
  const auto one_step = heat_eq.solve(mesh);

  const auto exact = exact_sol_vec(mesh, 0.01);

  for (std::size_t i = 0; i < one_step.size(); i++) {
    EXPECT_NEAR(one_step[i], exact[i], 1e-1);
  }
}

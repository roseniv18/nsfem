/**
 * ? TEST values and properties of the local velocity pressure matrix:
 * * Test entries for a known velocity-pressure coupling matrix
 */

#include <gtest/gtest.h>
#include <iostream>
#include "FEM/assemble/assemble.h"
#include "FEM/finite_element/P1_element.h"
#include "FEM/finite_element/P2_element.h"
#include "helpers/helpers.h"

using namespace nsfem::assembly;
using nsfem::fem::P1_FE, nsfem::fem::P2_FE, nsfem::fem::FEType;

/** The local velocity-pressure coupling matrix B (Stokes)
 * Test the entries of the matrix for:
 * Unit reference triangle with P2 velocity, P1 pressure
 * Standard basis ordering
 */
TEST(LocalVelocityPressureTest, HasExpectedEntries) {
  const std::vector<Node> nodes{{1, 0.0, 0.0}, {2, 1.0, 0.0}, {3, 0.0, 1.0},
                                {4, 0.5, 0.0}, {5, 0.5, 0.5}, {6, 0, 0.5}};

  const P1_FE pressure_fe;
  const P2_FE velocity_fe;

  const FEMap2D mapping(phys_element_tr6, nodes, velocity_fe);

  const auto B =
      gen_local_velocity_pressure_matr(mapping, velocity_fe, pressure_fe);

  const int n_vel = velocity_fe.get_ndofs();
  const int n_pre = pressure_fe.get_ndofs();

  ASSERT_EQ(n_vel, 6);
  ASSERT_EQ(n_pre, 3);
  ASSERT_EQ(B.rows(), 2 * n_vel);
  ASSERT_EQ(B.cols(), n_pre);

  MatrixXd expected(12, 3);

  expected.topRows(6) << -1.0 / 6.0, 0.0, 0.0, 0.0, 1.0 / 6.0, 0.0, 0.0, 0.0,
      0.0, 1.0 / 6.0, -1.0 / 6.0, 0.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 3.0,
      -1.0 / 6.0, -1.0 / 6.0, -1.0 / 3.0;

  expected.bottomRows(6) << -1.0 / 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      1.0 / 6.0, -1.0 / 6.0, -1.0 / 3.0, -1.0 / 6.0, 1.0 / 6.0, 1.0 / 3.0,
      1.0 / 6.0, 1.0 / 6.0, 0.0, -1.0 / 6.0;

  for (int i = 0; i < B.rows(); i++) {
    for (int j = 0; j < B.cols(); j++) {
      EXPECT_NEAR(B(i, j), expected(i, j), test_dtol);
    }
  }
}

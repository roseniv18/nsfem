/**
 * ? TEST values and properties of the local forcing load vector:
 * *
 */

#include <gtest/gtest.h>
#include <iostream>
#include "FEM/assemble/assemble.h"
#include "FEM/finite_element/P1_element.h"
#include "FEM/finite_element/P2_element.h"
#include "helpers/helpers.h"

using namespace nsfem::assembly;
using nsfem::fem::P1_FE, nsfem::fem::P2_FE, nsfem::fem::FEType;

/** The local forcing vector (Stokes)
 * Test the entries of the local vector for:
 * A constant force f = [1, 2]
 * The expected entries on the unit triangle are:
 * [0 0 0 1/6 1/6 1/6 0 0 0 1/3 1/3 1/3]
 */
TEST(LocalForcingVectorTest, HasExpectedEntries) {
  const std::vector<Node> nodes{{1, 0.0, 0.0}, {2, 1.0, 0.0}, {3, 0.0, 1.0},
                                {4, 0.5, 0.0}, {5, 0.5, 0.5}, {6, 0, 0.5}};

  const P2_FE velocity_fe;

  const FEMap2D mapping(phys_element_tr6, nodes, velocity_fe);

  const VecFunction forcing = [](double, double) {
    return Eigen::Vector2d(1.0, 2.0);
  };

  const VectorXd F =
      gen_local_stokes_forcing_vector(mapping, velocity_fe, forcing);

  const int n_vel = velocity_fe.get_ndofs();

  ASSERT_EQ(n_vel, 6);
  ASSERT_EQ(F.size(), 2 * n_vel);

  VectorXd expected(12);

  expected << 0.0, 0.0, 0.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 0.0, 0.0, 0.0,
      1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0;

  for (int i = 0; i < expected.size(); i++) {
    EXPECT_NEAR(F(i), expected(i), test_dtol);
  }
}

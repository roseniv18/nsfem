/**
 * ? TEST values and properties of the local velocity stiffness matrix:
 * * Test block-diagonal form and entries
 */

#include <gtest/gtest.h>
#include <iostream>
#include "FEM/assemble/assemble.h"
#include "FEM/finite_element/P1_element.h"
#include "FEM/finite_element/P2_element.h"
#include "helpers/helpers.h"

using namespace nsfem::assembly;
using nsfem::fem::P1_FE, nsfem::fem::P2_FE, nsfem::fem::FEType;

/** The local velocity stiffness matrix A (Stokes)
 * is a block-diagonal matrix of the form:
 * A = [[K 0], [0 K]], where K is the standard local stiffness matrix
 * for the Laplacian bilinear form. K is DOFxDOF (6x6 for P2 elements)
 */
TEST(LocalVelocityStiffnessTest, DiagonalBlocksAreScalarStiffnessMatrices) {
  const std::vector<Node> nodes{{1, 0.0, 0.0}, {2, 1.0, 0.0}, {3, 0.0, 1.0},
                                {4, 0.5, 0.0}, {5, 0.5, 0.5}, {6, 0, 0.5}};

  const P2_FE P2_element;

  const FEMap2D mapping(phys_element_tr6, nodes, P2_element);

  const auto ls_matrix = gen_local_stiffness_matr(mapping, P2_element);

  //   random viscosity value
  const double viscosity = 0.87;

  const auto lvs_matrix =
      gen_local_velocity_stiffness_matr(mapping, P2_element, viscosity);

  const int n = P2_element.get_ndofs();

  ASSERT_EQ(ls_matrix.rows(), n);
  ASSERT_EQ(ls_matrix.rows(), n);
  ASSERT_EQ(lvs_matrix.rows(), 2 * n);
  ASSERT_EQ(lvs_matrix.rows(), 2 * n);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      // Upper-left block
      EXPECT_NEAR(lvs_matrix(i, j), viscosity * ls_matrix(i, j), test_dtol);

      // Lower-right block
      EXPECT_NEAR(lvs_matrix(i + n, j + n), viscosity * ls_matrix(i, j),
                  test_dtol);

      // Off-diagonal blocks should be 0
      EXPECT_NEAR(lvs_matrix(i, n + j), 0.0, test_dtol);
      EXPECT_NEAR(lvs_matrix(n + i, j), 0.0, test_dtol);
    }
  }
}

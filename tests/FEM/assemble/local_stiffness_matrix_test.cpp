/**
 * ? TEST values and properties of the local stiffness matrix:
 * * Test value of the matrix on reference triangle
 * * Test value of the matrix on physical triangle
 * * Symmetry
 * * P2 local matrix dimensions
 */

#include <gtest/gtest.h>
#include <iostream>
#include "FEM/assemble/assemble.h"
#include "FEM/finite_element/P1_element.h"
#include "FEM/finite_element/P2_element.h"
#include "helpers/helpers.h"

/** Local stiffness matrix for reference triangle (P1)
 * Test if the local stiffness matrix for the reference triangle
 * is equal to [[1,-1/2,-1/2], [-1/2,1/2,0], [-1/2,0,1/2]]
 */
TEST(LocalStiffnessTest, ReferenceTriangle) {
  const std::vector<Node> nodes{{1, 0.0, 0.0}, {2, 1.0, 0.0}, {3, 0.0, 1.0}};

  const P1_FE P1_element;

  const FEMap2D ref_triangle(phys_element_tr3, nodes, P1_element);

  const auto ls_matrix = gen_local_stiffness_matr(ref_triangle, P1_element);

  EXPECT_NEAR(ls_matrix(0, 0), 1.0, test_dtol);
  EXPECT_NEAR(ls_matrix(1, 0), -0.5, test_dtol);
  EXPECT_NEAR(ls_matrix(2, 0), -0.5, test_dtol);
  EXPECT_NEAR(ls_matrix(0, 1), -0.5, test_dtol);
  EXPECT_NEAR(ls_matrix(1, 1), 0.5, test_dtol);
  EXPECT_NEAR(ls_matrix(2, 1), 0.0, test_dtol);
  EXPECT_NEAR(ls_matrix(0, 2), -0.5, test_dtol);
  EXPECT_NEAR(ls_matrix(1, 2), 0.0, test_dtol);
  EXPECT_NEAR(ls_matrix(2, 2), 0.5, test_dtol);
}

/** Local stiffness matrix for physical triangle (P1)
 * Test if the local stiffness matrix for the physical triangle
 * with nodes (1, 2), (4,2), (1, 6)
 * is equal to [[25/24, -2/3, -3/8], [-2/3, 2/3, 0], [-3/8, 0, 3/8]]
 */
TEST(LocalStiffnessTest, PhysTriangle) {
  const P1_FE P1_element;

  const FEMap2D mapping(phys_element_tr3, phys_nodes_tr3, P1_element);

  const auto phys_ls_matrix = gen_local_stiffness_matr(mapping, P1_element);

  EXPECT_NEAR(phys_ls_matrix(0, 0), 25.0 / 24.0, test_dtol);
  EXPECT_NEAR(phys_ls_matrix(1, 0), -2.0 / 3.0, test_dtol);
  EXPECT_NEAR(phys_ls_matrix(2, 0), -3.0 / 8.0, test_dtol);
  EXPECT_NEAR(phys_ls_matrix(0, 1), -2.0 / 3.0, test_dtol);
  EXPECT_NEAR(phys_ls_matrix(1, 1), 2.0 / 3.0, test_dtol);
  EXPECT_NEAR(phys_ls_matrix(2, 1), 0.0, test_dtol);
  EXPECT_NEAR(phys_ls_matrix(0, 2), -3.0 / 8.0, test_dtol);
  EXPECT_NEAR(phys_ls_matrix(1, 2), 0.0, test_dtol);
  EXPECT_NEAR(phys_ls_matrix(2, 2), 3.0 / 8.0, test_dtol);
}

/** Local stiffness matrix for reference triangle (P1)
 * Test if the local stiffness matrix for the reference triangle
 * is symmetric K=K^T
 */
TEST(LocalStiffnessTest, IsSymmetric) {
  const std::vector<Node> nodes{{1, 0.0, 0.0}, {2, 1.0, 0.0}, {3, 0.0, 1.0}};

  const P1_FE P1_element;

  const FEMap2D ref_triangle(phys_element_tr3, nodes, P1_element);

  const auto ls_matrix = gen_local_stiffness_matr(ref_triangle, P1_element);

  EXPECT_NEAR(ls_matrix(0, 1), ls_matrix(1, 0), test_dtol);
  EXPECT_NEAR(ls_matrix(0, 2), ls_matrix(2, 0), test_dtol);
  EXPECT_NEAR(ls_matrix(1, 2), ls_matrix(2, 1), test_dtol);
}

/** Local stiffness matrix for reference triangle
 * Test if the local stiffness matrix obtained from P2 elements
 * has appropriate number of DOFs
 */
TEST(LocalStiffnessTest, P2MatrixSize) {
  const std::vector<Node> nodes{
      {1, 0.0, 0.0},
      {2, 1.0, 0.0},
      {3, 0.0, 1.0},
  };

  const auto geo_fe = FE::build_fe_type(FEType::P1);
  const auto sol_fe = FE::build_fe_type(FEType::P2);

  const FEMap2D mapping(phys_element_tr3, nodes, *geo_fe);

  const auto K = gen_local_stiffness_matr(mapping, *sol_fe);

  EXPECT_EQ(K.rows(), 6);
  EXPECT_EQ(K.cols(), 6);
}

/** Local stiffness matrix for reference triangle (P2)
 * Test if the local stiffness matrix for the reference triangle
 * is symmetric M=M^T
 */
TEST(LocalStiffnessTest, IsSymmetricP2) {
  const std::vector<Node> nodes{{1, 0.0, 0.0}, {2, 1.0, 0.0}, {3, 0.0, 1.0}};

  const P1_FE geo_fe;
  const P2_FE sol_fe;

  const FEMap2D mapping(phys_element_tr3, nodes, geo_fe);

  const auto lm_matrix = gen_local_stiffness_matr(mapping, sol_fe);

  for (Eigen::Index i = 0; i < lm_matrix.rows(); i++) {
    for (Eigen::Index j = 0; j < lm_matrix.cols(); j++) {
      EXPECT_NEAR(lm_matrix(i, j), lm_matrix(j, i), test_dtol);
    }
  }
}

/** Local stiffness matrix for reference triangle (P2)
 * Test the entries of the matrix
 */
TEST(LocalStiffnessTEst, P2ReferenceTriangleEntryValues) {
  const std::vector<Node> nodes{{1, 0.0, 0.0}, {2, 1.0, 0.0}, {3, 0.0, 1.0}};

  const P1_FE geo_fe;
  const P2_FE sol_fe;

  const FEMap2D mapping(phys_element_tr3, nodes, geo_fe);

  const MatrixXd K = gen_local_stiffness_matr(mapping, sol_fe);

  MatrixXd expected(6, 6);
  expected << 1, 1.0 / 6.0, 1.0 / 6.0, -2.0 / 3.0, 0, -2.0 / 3.0, 1.0 / 6.0,
      0.5, 0, -2.0 / 3.0, 0, 0, 1.0 / 6.0, 0, 0.5, 0, 0, -2.0 / 3.0, -2.0 / 3.0,
      -2.0 / 3.0, 0, 8.0 / 3.0, -4.0 / 3.0, 0, 0, 0, 0, -4.0 / 3.0, 8.0 / 3.0,
      -4.0 / 3.0, -2.0 / 3.0, 0, -2.0 / 3.0, 0, -4.0 / 3.0, 8.0 / 3.0;

  ASSERT_EQ(K.rows(), 6);
  ASSERT_EQ(K.cols(), 6);

  for (Eigen::Index i = 0; i < expected.rows(); i++) {
    for (Eigen::Index j = 0; j < expected.cols(); j++) {
      EXPECT_NEAR(K(i, j), expected(i, j), test_dtol);
    }
  }
}

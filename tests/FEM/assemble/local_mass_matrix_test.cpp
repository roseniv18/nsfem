/**
 * ? TEST values and properties of the local mass matrix:
 * * Test value of the matrix on reference triangle
 * * Symmetry
 * * P2 local matrix dimensions
 */

#include <gtest/gtest.h>
#include "FEM/assemble/assemble.h"
#include "FEM/finite_element/P1_element.h"
#include "FEM/finite_element/P2_element.h"
#include "helpers/helpers.h"

/** Local mass matrix for reference triangle (P1)
 * Test if the local mass matrix for the reference triangle
 * is equal to [[1/12,1/24,1/24], [1/24,1/12,1/24], [1/24,1/24,1/12]]
 */
TEST(LocalMassTest, ReferenceTriangle) {
  const std::vector<Node> nodes{{1, 0.0, 0.0}, {2, 1.0, 0.0}, {3, 0.0, 1.0}};

  const P1_FE P1_element;

  const FEMap2D ref_triangle(phys_element_tr3, nodes, P1_element);

  const auto M = gen_local_mass_matr(ref_triangle, P1_element);

  EXPECT_NEAR(M(0, 0), 1.0 / 12.0, test_dtol);
  EXPECT_NEAR(M(1, 1), 1.0 / 12.0, test_dtol);
  EXPECT_NEAR(M(2, 2), 1.0 / 12.0, test_dtol);

  EXPECT_NEAR(M(0, 1), 1.0 / 24.0, test_dtol);
  EXPECT_NEAR(M(0, 2), 1.0 / 24.0, test_dtol);
  EXPECT_NEAR(M(1, 0), 1.0 / 24.0, test_dtol);
  EXPECT_NEAR(M(1, 2), 1.0 / 24.0, test_dtol);
  EXPECT_NEAR(M(2, 0), 1.0 / 24.0, test_dtol);
  EXPECT_NEAR(M(2, 1), 1.0 / 24.0, test_dtol);
}

/**
 * P2 Mass matrix on Reference triangle
 *
 */
TEST(LocalMassTest, P2ReferenceTriangleEntryValues) {
  const std::vector<Node> nodes{{1, 0.0, 0.0}, {2, 1.0, 0.0}, {3, 0.0, 1.0}};

  const P1_FE geo_fe;
  const P2_FE sol_fe;
  const FEMap2D mapping(phys_element_tr3, nodes, geo_fe);

  const MatrixXd M = gen_local_mass_matr(mapping, sol_fe);

  MatrixXd expected(6, 6);
  expected << 6, -1, -1, 0, -4, 0, -1, 6, -1, 0, 0, -4, -1, -1, 6, -4, 0, 0, 0,
      0, -4, 32, 16, 16, -4, 0, 0, 16, 32, 16, 0, -4, 0, 16, 16, 32;

  expected /= 360.0;

  ASSERT_EQ(M.rows(), 6);
  ASSERT_EQ(M.cols(), 6);

  for (Eigen::Index i = 0; i < expected.rows(); i++) {
    for (Eigen::Index j = 0; j < expected.cols(); j++) {
      EXPECT_NEAR(M(i, j), expected(i, j), test_dtol);
    }
  }
}

/** Local mass matrix for reference triangle (P1)
 * Test if the local mass matrix for the reference triangle
 * is symmetric M=M^T
 */
TEST(LocalMassTest, IsSymmetric) {
  const std::vector<Node> nodes{{1, 0.0, 0.0}, {2, 1.0, 0.0}, {3, 0.0, 1.0}};

  const P1_FE P1_element;

  const FEMap2D ref_triangle(phys_element_tr3, nodes, P1_element);

  const auto M = gen_local_mass_matr(ref_triangle, P1_element);

  EXPECT_NEAR(M(0, 1), M(1, 0), test_dtol);
  EXPECT_NEAR(M(0, 2), M(2, 0), test_dtol);
  EXPECT_NEAR(M(1, 2), M(2, 1), test_dtol);
}

/** Local mass matrix for reference triangle (P2)
 * Test if the local mass matrix obtained from P2 elements
 * has appropriate number of DOFs
 */
TEST(LocalMassTest, P2MatrixSize) {
  const std::vector<Node> nodes{
      {1, 0.0, 0.0},
      {2, 1.0, 0.0},
      {3, 0.0, 1.0},
  };

  const auto geo_fe = FE::build_fe_type(FEType::P1);
  const auto sol_fe = FE::build_fe_type(FEType::P2);

  const FEMap2D mapping(phys_element_tr3, nodes, *geo_fe);

  const auto M = gen_local_mass_matr(mapping, *sol_fe);

  EXPECT_EQ(M.rows(), 6);
  EXPECT_EQ(M.cols(), 6);
}

/** Local mass matrix for reference triangle (P2)
 * Test if the local mass matrix for the reference triangle
 * is symmetric M=M^T
 */
TEST(LocalMassTest, IsSymmetricP2) {
  const std::vector<Node> nodes{{1, 0.0, 0.0}, {2, 1.0, 0.0}, {3, 0.0, 1.0}};

  const auto geo_fe = FE::build_fe_type(FEType::P1);
  const auto sol_fe = FE::build_fe_type(FEType::P2);

  const FEMap2D mapping(phys_element_tr3, nodes, *geo_fe);

  const auto M = gen_local_mass_matr(mapping, *sol_fe);

  for (Eigen::Index i = 0; i < M.rows(); i++) {
    for (Eigen::Index j = 0; j < M.cols(); j++) {
      EXPECT_NEAR(M(i, j), M(j, i), test_dtol);
    }
  }
}

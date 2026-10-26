/**
 * ? TEST different combinations of the geometry space + solution space:
 * * Isoparametric: 		GEO = SOL 	(Triangle3 + P1, Triangle6 + P2)
 * * Subparametric: 		GEO < SOL 	(Triangle3 + P2)
 * * Superparametric: 		GEO > SOL 	(Triangle6 + P1)
 */

#include <gtest/gtest.h>
#include "FEM/assemble/assemble.h"
#include "FEM/finite_element/P1_element.h"
#include "FEM/finite_element/P2_element.h"
#include "FEM/mapping/fe_mapping.h"
#include "helpers/helpers.h"

using Eigen::VectorXd, Eigen::Matrix2d, Eigen::MatrixXd;
using namespace nsfem::assembly;
using nsfem::fem::P1_FE, nsfem::fem::P2_FE, nsfem::fem::FEType;

/* ISOPARAMETRIC P1 + P1 */
TEST(MixedOrderAssemblyTest, P1GeoP1Sol) {
  const P1_FE geo_fe;
  const P1_FE sol_fe;

  /* Geometry is linear / P1 */
  const FEMap2D mapping(phys_element_tr3, phys_nodes_tr3, geo_fe);

  /* Solution is described with linear / P1 elements */
  const MatrixXd M_local = gen_local_mass_matr(mapping, sol_fe);

  //   3 DOFs -> local matrix should be 3x3
  ASSERT_EQ(M_local.rows(), 3);
  ASSERT_EQ(M_local.cols(), 3);

  /**
   * For standard nodal Lagrange elements,
   * the sum of the entries of the local mass matrix
   * is equal to the area / volume of the element
   * In this case, the element's area is 6
   */
  EXPECT_NEAR(M_local.sum(), 6.0, test_dtol);

  /* Symmetry */
  EXPECT_TRUE(M_local.isApprox(M_local.transpose()));
}

/* ISOPARAMETRIC P2 + P2 */
TEST(MixedOrderAssemblyTest, P2GeoP2Sol) {
  const P2_FE geo_fe;
  const P2_FE sol_fe;

  /**
   * Geometry is quadratic / P2
   * Note: Use nodes that describe a triangle with curved edge(s). This results
   * in a nonlinear mapping */
  const FEMap2D mapping(phys_element_tr6, phys_nodes_tr6_nonlinear, geo_fe);

  /* Solution is described with quadratic / P2 elements */
  const MatrixXd M_local = gen_local_mass_matr(mapping, sol_fe);

  //   6 DOFs -> local matrix should be 6x6
  ASSERT_EQ(M_local.rows(), 6);
  ASSERT_EQ(M_local.cols(), 6);

  /**
   * For standard nodal Lagrange elements,
   * the sum of the entries of the local mass matrix
   * is equal to the area / volume of the element
   * In this case, the element's area is 25/3
   */
  EXPECT_NEAR(M_local.sum(), 25.0 / 3.0, test_dtol);

  /* Symmetry */
  EXPECT_TRUE(M_local.isApprox(M_local.transpose()));
}

/* SUBPARAMETRIC */
TEST(MixedOrderAssemblyTest, P1GeoP2Sol) {
  const P1_FE geo_fe;
  const P2_FE sol_fe;

  /* Geometry is linear / P1 */
  const FEMap2D mapping(phys_element_tr3, phys_nodes_tr3, geo_fe);

  /* Solution is described with quadratic / P2 elements */
  const MatrixXd M_local = gen_local_mass_matr(mapping, sol_fe);

  //   6 DOFs -> local matrix should be 6x6
  ASSERT_EQ(M_local.rows(), 6);
  ASSERT_EQ(M_local.cols(), 6);

  /**
   * For standard nodal Lagrange elements,
   * the sum of the entries of the local mass matrix
   * is equal to the area / volume of the element
   * In this case, the element's area is 6
   */
  EXPECT_NEAR(M_local.sum(), 6.0, test_dtol);

  /* Symmetry */
  EXPECT_TRUE(M_local.isApprox(M_local.transpose()));
}

/* SUPERPARAMETRIC */
TEST(MixedOrderAssemblyTest, P2GeoP1Sol) {
  const P2_FE geo_fe;
  const P1_FE sol_fe;

  /**
   * Geometry is quadratic / P2
   * Note: Use nodes that describe a triangle with curved edge(s). This results
   * in a nonlinear mapping */
  const FEMap2D mapping(phys_element_tr6, phys_nodes_tr6_nonlinear, geo_fe);

  /* Solution is described with linear / P1 elements */
  const MatrixXd M_local = gen_local_mass_matr(mapping, sol_fe);

  //   3 DOFs -> local matrix should be 3x3
  ASSERT_EQ(M_local.rows(), 3);
  ASSERT_EQ(M_local.cols(), 3);

  /**
   * For standard nodal Lagrange elements,
   * the sum of the entries of the local mass matrix
   * is equal to the area / volume of the element
   * In this case, the element's area is 25/3
   */
  EXPECT_NEAR(M_local.sum(), 25.0 / 3.0, test_dtol);

  /* Symmetry */
  EXPECT_TRUE(M_local.isApprox(M_local.transpose()));
}

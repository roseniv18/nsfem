#include <gtest/gtest.h>
#include "FEM/assemble/assemble.h"
#include "FEM/finite_element/P1_element.h"
#include "helpers/helpers.h"

/** Local stiffness matrix for reference triangle
 * Test if the local stiffness matrix for the reference triangle
 * is equal to [[1,-1/2,-1/2], [-1/2,1/2,0], [-1/2,0,1/2]]
 */
TEST(LocalStiffnessTest, ReferenceTriangle) {
  std::vector<Node> nodes{{1, 0.0, 0.0}, {2, 1.0, 0.0}, {3, 0.0, 1.0}};
  Element element{.dim = 2,
                  .element_tag = 1,
                  .type = ElementType::Triangle3,
                  .physical_tags = {},
                  .node_ids = {0, 1, 2}};

  const P1_FE P1_element;

  const FEMap2D ref_triangle(element, nodes, P1_element);

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

/** Local stiffness matrix for physical triangle
 * Test if the local stiffness matrix for the physical triangle
 * with nodes (1, 2), (4,2), (1, 6)
 * is equal to [[25/24, -2/3, -3/8], [-2/3, 2/3, 0], [-3/8, 0, 3/8]]
 */
TEST(LocalStiffnessTest, PhysTriangle) {
  std::vector<Node> phys_nodes{{1, 1.0, 2.0}, {2, 4.0, 2.0}, {3, 1.0, 6.0}};
  Element phys_element{.dim = 2,
                       .element_tag = 1,
                       .type = ElementType::Triangle3,
                       .physical_tags = {},
                       .node_ids = {0, 1, 2}};

  const P1_FE P1_element;

  const FEMap2D mapping(phys_element, phys_nodes, P1_element);

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

/** Local stiffness matrix for reference triangle
 * Test if the local stiffness matrix for the reference triangle
 * is symmetric K=K^T
 */
TEST(LocalStiffnessTest, IsSymmetric) {
  std::vector<Node> nodes{{1, 0.0, 0.0}, {2, 1.0, 0.0}, {3, 0.0, 1.0}};
  Element element{.dim = 2,
                  .element_tag = 1,
                  .type = ElementType::Triangle3,
                  .physical_tags = {},
                  .node_ids = {0, 1, 2}};

  const P1_FE P1_element;

  const FEMap2D ref_triangle(element, nodes, P1_element);

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
  std::vector<Node> nodes{
      {1, 0.0, 0.0},
      {2, 1.0, 0.0},
      {3, 0.0, 1.0},
  };

  Element element{
      .dim = 2,
      .element_tag = 1,
      .type = ElementType::Triangle3,
      .physical_tags = {},
      .node_ids = {0, 1, 2},
  };

  const auto fe = FE::build_fe_type(FEType::P2);

  const FEMap2D mapping(element, nodes, *fe);

  const auto K = gen_local_stiffness_matr(mapping, *fe);

  EXPECT_EQ(K.rows(), 6);
  EXPECT_EQ(K.cols(), 6);
}

/** Local mass matrix for reference triangle
 * Test if the local mass matrix for the reference triangle
 * is symmetric M=M^T
 */
TEST(LocalStiffnessTest, IsSymmetricP2) {
  std::vector<Node> nodes{{1, 0.0, 0.0}, {2, 1.0, 0.0}, {3, 0.0, 1.0}};
  Element element{.dim = 2,
                  .element_tag = 1,
                  .type = ElementType::Triangle3,
                  .physical_tags = {},
                  .node_ids = {0, 1, 2}};

  const std::unique_ptr<FE> fe = FE::build_fe_type(FEType::P2);

  const FEMap2D ref_triangle(element, nodes, *fe);

  const auto lm_matrix = gen_local_stiffness_matr(ref_triangle, *fe);

  for (std::size_t i = 0; i < lm_matrix.rows(); i++) {
    for (std::size_t j = 0; j < lm_matrix.cols(); j++) {
      EXPECT_NEAR(lm_matrix(i, j), lm_matrix(j, i), test_dtol);
    }
  }
}

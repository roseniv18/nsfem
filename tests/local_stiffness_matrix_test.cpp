#include <gtest/gtest.h>
#include "assemble/assemble.h"

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
                  .node_indices = {0, 1, 2}};

  const TriangleGEO ref_triangle(element, nodes);

  const auto ls_matrix = gen_local_stiffness_matr(ref_triangle);

  EXPECT_DOUBLE_EQ(ls_matrix(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(ls_matrix(1, 0), -0.5);
  EXPECT_DOUBLE_EQ(ls_matrix(2, 0), -0.5);
  EXPECT_DOUBLE_EQ(ls_matrix(0, 1), -0.5);
  EXPECT_DOUBLE_EQ(ls_matrix(1, 1), 0.5);
  EXPECT_DOUBLE_EQ(ls_matrix(2, 1), 0.0);
  EXPECT_DOUBLE_EQ(ls_matrix(0, 2), -0.5);
  EXPECT_DOUBLE_EQ(ls_matrix(1, 2), 0.0);
  EXPECT_DOUBLE_EQ(ls_matrix(2, 2), 0.5);
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
                       .node_indices = {0, 1, 2}};

  const TriangleGEO phys_triangle(phys_element, phys_nodes);

  const auto phys_ls_matrix = gen_local_stiffness_matr(phys_triangle);

  EXPECT_NEAR(phys_ls_matrix(0, 0), 25.0 / 24.0, 1e-12);
  EXPECT_NEAR(phys_ls_matrix(1, 0), -2.0 / 3.0, 1e-12);
  EXPECT_NEAR(phys_ls_matrix(2, 0), -3.0 / 8.0, 1e-12);
  EXPECT_NEAR(phys_ls_matrix(0, 1), -2.0 / 3.0, 1e-12);
  EXPECT_NEAR(phys_ls_matrix(1, 1), 2.0 / 3.0, 1e-12);
  EXPECT_NEAR(phys_ls_matrix(2, 1), 0.0, 1e-12);
  EXPECT_NEAR(phys_ls_matrix(0, 2), -3.0 / 8.0, 1e-12);
  EXPECT_NEAR(phys_ls_matrix(1, 2), 0.0, 1e-12);
  EXPECT_NEAR(phys_ls_matrix(2, 2), 3.0 / 8.0, 1e-12);
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
                  .node_indices = {0, 1, 2}};

  const TriangleGEO ref_triangle(element, nodes);

  const auto ls_matrix = gen_local_stiffness_matr(ref_triangle);

  EXPECT_NEAR(ls_matrix(0, 1), ls_matrix(1, 0), 1e-12);
  EXPECT_NEAR(ls_matrix(0, 2), ls_matrix(2, 0), 1e-12);
  EXPECT_NEAR(ls_matrix(1, 2), ls_matrix(2, 1), 1e-12);
}

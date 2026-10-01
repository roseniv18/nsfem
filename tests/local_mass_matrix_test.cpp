#include <gtest/gtest.h>
#include "FEM/assemble/assemble.h"
#include "FEM/finite_element/P1_element.h"

/** Local mass matrix for reference triangle
 * Test if the local mass matrix for the reference triangle
 * is equal to [[1/12,1/24,1/24], [1/24,1/12,1/24], [1/24,1/24,1/12]]
 */
TEST(LocalMassTest, ReferenceTriangle) {
  std::vector<Node> nodes{{1, 0.0, 0.0}, {2, 1.0, 0.0}, {3, 0.0, 1.0}};
  Element element{.dim = 2,
                  .element_tag = 1,
                  .type = ElementType::Triangle3,
                  .physical_tags = {},
                  .node_indices = {0, 1, 2}};

  const P1_FE P1_element;

  const FEMap2D ref_triangle(element, nodes, P1_element);

  const auto lm_matrix = gen_local_mass_matr(ref_triangle, P1_element);
  EXPECT_NEAR(lm_matrix(0, 0), 1.0 / 12.0, 1e-12);
  EXPECT_NEAR(lm_matrix(1, 1), 1.0 / 12.0, 1e-12);
  EXPECT_NEAR(lm_matrix(2, 2), 1.0 / 12.0, 1e-12);

  EXPECT_NEAR(lm_matrix(0, 1), 1.0 / 24.0, 1e-12);
  EXPECT_NEAR(lm_matrix(0, 2), 1.0 / 24.0, 1e-12);
  EXPECT_NEAR(lm_matrix(1, 0), 1.0 / 24.0, 1e-12);
  EXPECT_NEAR(lm_matrix(1, 2), 1.0 / 24.0, 1e-12);
  EXPECT_NEAR(lm_matrix(2, 0), 1.0 / 24.0, 1e-12);
  EXPECT_NEAR(lm_matrix(2, 1), 1.0 / 24.0, 1e-12);
}

/** Local mass matrix for reference triangle
 * Test if the local mass matrix for the reference triangle
 * is symmetric M=M^T
 */
TEST(LocalMassTest, IsSymmetric) {
  std::vector<Node> nodes{{1, 0.0, 0.0}, {2, 1.0, 0.0}, {3, 0.0, 1.0}};
  Element element{.dim = 2,
                  .element_tag = 1,
                  .type = ElementType::Triangle3,
                  .physical_tags = {},
                  .node_indices = {0, 1, 2}};

  const P1_FE P1_element;

  const FEMap2D ref_triangle(element, nodes, P1_element);

  const auto lm_matrix = gen_local_mass_matr(ref_triangle, P1_element);

  EXPECT_NEAR(lm_matrix(0, 1), lm_matrix(1, 0), 1e-12);
  EXPECT_NEAR(lm_matrix(0, 2), lm_matrix(2, 0), 1e-12);
  EXPECT_NEAR(lm_matrix(1, 2), lm_matrix(2, 1), 1e-12);
}

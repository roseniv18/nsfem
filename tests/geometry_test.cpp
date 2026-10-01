#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "FEM/finite_element/P1_element.h"
#include "FEM/mapping/fe_mapping.h"

using Eigen::VectorXd;

/** Jacobian Reference Triangle
 * Test if the Jacobian matrix for the reference triangle
 * is the identity matrix [[1,0],[0,1]]
 */
TEST(FEMap2DTest, JacobianForReferenceTriangle) {
  std::vector<Node> nodes{{1, 0.0, 0.0}, {2, 1.0, 0.0}, {3, 0.0, 1.0}};
  Element element{.dim = 2,
                  .element_tag = 1,
                  .type = ElementType::Triangle3,
                  .physical_tags = {},
                  .node_indices = {0, 1, 2}};

  const P1_FE P1_element;

  const FEMap2D mapping(element, nodes, P1_element);

  const auto J = mapping.jacobian();

  EXPECT_DOUBLE_EQ(J(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(J(1, 0), 0.0);
  EXPECT_DOUBLE_EQ(J(0, 1), 0.0);
  EXPECT_DOUBLE_EQ(J(1, 1), 1.0);
}

/** DET Jacobian Reference Triangle
 * Test if the Determinant of the Jacobian matrix for the reference triangle
 * is 1
 */
TEST(FEMap2DTest, DetJacobianForReferenceTriangle) {
  std::vector<Node> nodes{{1, 0.0, 0.0}, {2, 1.0, 0.0}, {3, 0.0, 1.0}};
  Element element{.dim = 2,
                  .element_tag = 1,
                  .type = ElementType::Triangle3,
                  .physical_tags = {},
                  .node_indices = {0, 1, 2}};

  const P1_FE P1_element;

  const FEMap2D ref_triangle(element, nodes, P1_element);

  const auto detJ = ref_triangle.det_jacobian();

  EXPECT_DOUBLE_EQ(detJ, 1.0);
}

/** Jacobian Physical Triangle
 * Test if the Jacobian matrix for the physical triangle
 * with nodes (1,2), (4,2), (1,6)
 * is the matrix [[3,0],[0,4]]
 */
TEST(FEMap2DTest, JacobianForPhysicalTriangle) {
  std::vector<Node> nodes{{1, 1.0, 2.0}, {2, 4.0, 2.0}, {3, 1.0, 6.0}};
  Element element{.dim = 2,
                  .element_tag = 1,
                  .type = ElementType::Triangle3,
                  .physical_tags = {},
                  .node_indices = {0, 1, 2}};

  const P1_FE P1_element;

  const FEMap2D phys_triangle(element, nodes, P1_element);

  const auto J = phys_triangle.jacobian();

  EXPECT_DOUBLE_EQ(J(0, 0), 3.0);
  EXPECT_DOUBLE_EQ(J(1, 0), 0.0);
  EXPECT_DOUBLE_EQ(J(0, 1), 0.0);
  EXPECT_DOUBLE_EQ(J(1, 1), 4.0);
}

/** DET Jacobian Physical Triangle
 * Test if the Determinant of the Jacobian matrix for the physical triangle
 * with nodes (1,2), (4,2), (1,6)
 * is 12
 */
TEST(FEMap2DTest, DetJacobianForPhysicalTriangle) {
  std::vector<Node> nodes{{1, 1.0, 2.0}, {2, 4.0, 2.0}, {3, 1.0, 6.0}};
  const Element element{.dim = 2,
                        .element_tag = 1,
                        .type = ElementType::Triangle3,
                        .physical_tags = {},
                        .node_indices = {0, 1, 2}};

  const P1_FE P1_element;

  const FEMap2D phys_triangle(element, nodes, P1_element);

  const auto detJ = phys_triangle.det_jacobian();

  EXPECT_DOUBLE_EQ(detJ, 12.0);
}

/** Inverse Jacobian Transpose Physical Triangle
 * Test if the inverse Jacobian matrix transposed for the physical triangle
 * with nodes (1,2), (4,2), (1,6)
 * is [[1/3,0],[0,1/4]]
 */
TEST(FEMap2DTest, InvJacobianTForPhysicalTriangle) {
  std::vector<Node> nodes{{1, 1.0, 2.0}, {2, 4.0, 2.0}, {3, 1.0, 6.0}};
  const Element element{.dim = 2,
                        .element_tag = 1,
                        .type = ElementType::Triangle3,
                        .physical_tags = {},
                        .node_indices = {0, 1, 2}};

  const P1_FE P1_element;

  const FEMap2D phys_triangle(element, nodes, P1_element);

  const auto JinvT = phys_triangle.jacobianInvT();

  EXPECT_NEAR(JinvT(0, 0), 1.0 / 3.0, 1e-12);
  EXPECT_NEAR(JinvT(1, 0), 0.0, 1e-12);
  EXPECT_NEAR(JinvT(0, 1), 0.0, 1e-12);
  EXPECT_NEAR(JinvT(1, 1), 1.0 / 4.0, 1e-12);
}

/** Compute Physical Gradients
 * Test if the physical gradients for the triangle
 * with nodes (1,2), (4,2), (1,6)
 * are [-1/3, -1/4], [1/3, 0], [0, 1/4]
 */
TEST(FEMap2DTest, ComputesPhysicalGradients) {
  std::vector<Node> nodes{{1, 1.0, 2.0}, {2, 4.0, 2.0}, {3, 1.0, 6.0}};
  const Element element{.dim = 2,
                        .element_tag = 1,
                        .type = ElementType::Triangle3,
                        .physical_tags = {},
                        .node_indices = {0, 1, 2}};

  const P1_FE P1_element;

  const FEMap2D phys_triangle(element, nodes, P1_element);

  const auto grads = phys_triangle.get_phys_grads();

  EXPECT_NEAR(grads[0].x, -1.0 / 3.0, 1e-12);
  EXPECT_NEAR(grads[0].y, -1.0 / 4.0, 1e-12);

  EXPECT_NEAR(grads[1].x, 1.0 / 3.0, 1e-12);
  EXPECT_NEAR(grads[1].y, 0.0, 1e-12);

  EXPECT_NEAR(grads[2].x, 0.0, 1e-12);
  EXPECT_NEAR(grads[2].y, 1.0 / 4.0, 1e-12);
}

/** Reject Non-Triangle Element
 * Test if the FEMap2D constructor will throw
 * if we pass a non-triangle element
 */
TEST(FEMap2DTest, RejectsNonTriangleElement) {
  std::vector<Node> nodes{{1, 0.0, 0.0}, {2, 1.0, 0.0}, {3, 0.0, 1.0}};
  //   set dim to 1 (1D)
  const Element element{.dim = 1,
                        .element_tag = 1,
                        .type = ElementType::Line2,
                        .physical_tags = {},
                        .node_indices = {0, 1}};

  const P1_FE P1_element;

  EXPECT_THROW(FEMap2D mapping(element, nodes, P1_element),
               std::invalid_argument);
}

/** Reject Degenerate Triangle (detJ ~= 0)
 * Test if the FEMap2D constructor will throw
 * if we pass a degenerate element
 */
TEST(FEMap2DTest, RejectsDegenerateTriangle) {
  // coordinates are collinear
  std::vector<Node> nodes{{1, 0.0, 0.0}, {2, 1.0, 0.0}, {3, 2.0, 0.0}};
  const Element element{.dim = 2,
                        .element_tag = 1,
                        .type = ElementType::Triangle3,
                        .physical_tags = {},
                        .node_indices = {0, 1, 2}};

  const P1_FE P1_element;

  EXPECT_THROW(FEMap2D mapping(element, nodes, P1_element),
               std::invalid_argument);
}

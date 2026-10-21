#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "FEM/finite_element/P1_element.h"
#include "FEM/finite_element/P2_element.h"
#include "FEM/mapping/fe_mapping.h"
#include "helpers/helpers.h"

using Eigen::VectorXd, Eigen::Matrix2d;

const std::vector<Node> phys_nodes_tr3{{1, 1.0, 2.0},
                                       {2, 4.0, 2.0},
                                       {3, 1.0, 6.0}};

const std::vector<Node> phys_nodes_tr6_linear{{1, 1.0, 2.0}, {2, 4.0, 2.0},
                                              {3, 1.0, 6.0}, {4, 2.5, 2.0},
                                              {5, 2.5, 4.0}, {6, 1.0, 4.0}};

const std::vector<Node> phys_nodes_tr6_nonlinear{{1, 1.0, 2.0}, {2, 4.0, 2.0},
                                                 {3, 1.0, 6.0}, {4, 2.5, 2.0},
                                                 {5, 3.0, 4.5}, {6, 1.0, 4.0}};

const Element phys_element_tr3{.dim = 2,
                               .element_tag = 1,
                               .type = ElementType::Triangle3,
                               .physical_tags = {},
                               .node_ids = {0, 1, 2},
                               .edge_ids = {0, 1, 2}};

const Element phys_element_tr6{.dim = 2,
                               .element_tag = 1,
                               .type = ElementType::Triangle6,
                               .physical_tags = {},
                               .node_ids = {0, 1, 2, 3, 4, 5},
                               .edge_ids = {0, 1, 2}};

/** Physical point mapping for linear Triangle3 geometry
 * x(xi, eta) = (1 + 3*xi, 2 + 4*eta)
 * Check the mapping [ref -> phys]:
 * (0.2, 0.3) -> (1.6, 3.2)
 */
TEST(FEMap2DTest, PhysicalPointTriangle3) {
  const P1_FE P1_element;
  const FEMap2D mapping(phys_element_tr3, phys_nodes_tr3, P1_element);

  const RefNode ref_pt{0.2, 0.3};
  const PhysNode phys_pt = mapping.physical_point(ref_pt);

  EXPECT_NEAR(phys_pt.x, 1.6, test_dtol);
  EXPECT_NEAR(phys_pt.y, 3.2, test_dtol);
}

/** Physical point mapping for nonlinear Triangle6 geometry
 * x(xi, eta) = (1 + 3*xi + 2*xi*eta, 2 + 4*eta +2*xi*eta)
 * Check the mapping [ref -> phys]:
 * (0.2, 0.3) -> (1.72, 3.32)
 */
TEST(FEMap2DTest, PhysicalPointTriangle6Nonlinear) {
  const P2_FE P2_element;
  const FEMap2D mapping(phys_element_tr6, phys_nodes_tr6_nonlinear, P2_element);

  const RefNode ref_pt{0.2, 0.3};
  const PhysNode phys_pt = mapping.physical_point(ref_pt);

  EXPECT_NEAR(phys_pt.x, 1.72, test_dtol);
  EXPECT_NEAR(phys_pt.y, 3.32, test_dtol);
}

/** Jacobian Reference Triangle
 * Test if the Jacobian matrix for the reference P1 triangle
 * is the identity matrix [[1,0],[0,1]]
 */
TEST(FEMap2DTest, JacobianForReferenceTriangle3) {
  std::vector<Node> nodes{{1, 0.0, 0.0}, {2, 1.0, 0.0}, {3, 0.0, 1.0}};

  const P1_FE P1_element;

  const FEMap2D mapping(phys_element_tr3, nodes, P1_element);

  const auto J = mapping.jacobian(RefNode{1.0, 0.0});

  EXPECT_DOUBLE_EQ(J(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(J(1, 0), 0.0);
  EXPECT_DOUBLE_EQ(J(0, 1), 0.0);
  EXPECT_DOUBLE_EQ(J(1, 1), 1.0);
}

/** Nodal Interpolation for Triangle3 geometry
 * Each reference geometry node must map to its corresponding physical node
 */
TEST(FEMap2DTest, GeometryNodeInterpolationTriangle3) {
  const P1_FE geometry_fe;
  const FEMap2D mapping(phys_element_tr3, phys_nodes_tr3, geometry_fe);
  const auto ref_nodes = geometry_fe.get_ref_nodes();

  ASSERT_EQ(ref_nodes.size(), phys_nodes_tr3.size());

  for (std::size_t i = 0; i < ref_nodes.size(); i++) {
    const PhysNode phys_pt = mapping.physical_point(ref_nodes.at(i));

    EXPECT_NEAR(phys_pt.x, phys_nodes_tr3.at(i).x, test_dtol);
    EXPECT_NEAR(phys_pt.y, phys_nodes_tr3.at(i).y, test_dtol);
  }
}

/** Nodal Interpolation for Triangle6 geometry (Nonlinear)
 * Each reference geometry node must map to its corresponding physical node
 */
TEST(FEMap2DTest, GeometryNodeInterpolationTriangle6Nonlinear) {
  const P2_FE geometry_fe;
  const FEMap2D mapping(phys_element_tr6, phys_nodes_tr6_nonlinear,
                        geometry_fe);
  const auto ref_nodes = geometry_fe.get_ref_nodes();

  ASSERT_EQ(ref_nodes.size(), phys_nodes_tr6_nonlinear.size());

  for (std::size_t i = 0; i < ref_nodes.size(); i++) {
    const PhysNode phys_pt = mapping.physical_point(ref_nodes.at(i));

    EXPECT_NEAR(phys_pt.x, phys_nodes_tr6_nonlinear.at(i).x, test_dtol);
    EXPECT_NEAR(phys_pt.y, phys_nodes_tr6_nonlinear.at(i).y, test_dtol);
  }
}

/** Jacobian Reference Triangle
 * Test if the Jacobian matrix for the reference P2 triangle
 * is the identity matrix [[1,0],[0,1]]
 */
TEST(FEMap2DTest, JacobianForReferenceTriangle6) {
  std::vector<Node> nodes{{1, 0.0, 0.0}, {2, 1.0, 0.0}, {3, 0.0, 1.0},
                          {4, 0.5, 0.0}, {5, 0.5, 0.5}, {6, 0.0, 0.5}};

  const P2_FE P2_element;

  const FEMap2D mapping(phys_element_tr6, nodes, P2_element);

  const auto J = mapping.jacobian(RefNode{1.0, 0.0});

  EXPECT_DOUBLE_EQ(J(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(J(1, 0), 0.0);
  EXPECT_DOUBLE_EQ(J(0, 1), 0.0);
  EXPECT_DOUBLE_EQ(J(1, 1), 1.0);
}

/** DET Jacobian Reference Triangle
 * Test if the Determinant of the Jacobian matrix for the P1 reference triangle
 * is 1
 */
TEST(FEMap2DTest, DetJacobianForReferenceTriangle3) {
  std::vector<Node> nodes{{1, 0.0, 0.0}, {2, 1.0, 0.0}, {3, 0.0, 1.0}};

  const P1_FE P1_element;

  const FEMap2D ref_triangle(phys_element_tr3, nodes, P1_element);

  const auto detJ = ref_triangle.det_jacobian(RefNode{1.0, 0.0});

  EXPECT_DOUBLE_EQ(detJ, 1.0);
}

/** DET Jacobian Reference Triangle
 * Test if the Determinant of the Jacobian matrix for the P2 reference triangle
 * is 1
 */
TEST(FEMap2DTest, DetJacobianForReferenceTriangle6) {
  std::vector<Node> nodes{{1, 0.0, 0.0}, {2, 1.0, 0.0}, {3, 0.0, 1.0},
                          {4, 0.5, 0.0}, {5, 0.5, 0.5}, {6, 0.0, 0.5}};

  const P2_FE P2_element;

  const FEMap2D ref_triangle(phys_element_tr6, nodes, P2_element);

  const auto detJ = ref_triangle.det_jacobian(RefNode{1.0, 0.0});

  EXPECT_DOUBLE_EQ(detJ, 1.0);
}

/** Jacobian Physical Triangle
 * LINEAR ISOPARAMETRIC MAPPING  [Triangle3] -> [P1]
 * Test if the Jacobian matrix for the physical triangle
 * with nodes (1,2), (4,2), (1,6)
 * is the matrix [[3,0],[0,4]]
 */
TEST(FEMap2DTest, JacobianForPhysicalTriangle3) {
  const P1_FE P1_element;

  const FEMap2D phys_triangle(phys_element_tr3, phys_nodes_tr3, P1_element);

  const auto J = phys_triangle.jacobian(RefNode{1.0, 0.0});

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
  const P1_FE P1_element;

  const FEMap2D phys_triangle(phys_element_tr3, phys_nodes_tr3, P1_element);

  const auto detJ = phys_triangle.det_jacobian(RefNode{1.0, 0.0});

  EXPECT_DOUBLE_EQ(detJ, 12.0);
}

/** Jacobian Physical Triangle
 * LINEAR ISOPARAMETRIC MAPPING  [Triangle6] -> [P2]
 * Test if the Jacobian matrix for the physical triangle
 * with nodes (1,2), (4,2), (1,6), (2.5, 2), (2.5, 4), (1, 4)
 * is the matrix [[3,0],[0,4]]
 */
TEST(FEMap2DTest, JacobianForPhysicalTriangle6Linear) {
  const P2_FE P2_element;

  const FEMap2D phys_triangle(phys_element_tr6, phys_nodes_tr6_linear,
                              P2_element);

  const auto J = phys_triangle.jacobian(RefNode{1.0, 0.0});

  EXPECT_DOUBLE_EQ(J(0, 0), 3.0);
  EXPECT_DOUBLE_EQ(J(1, 0), 0.0);
  EXPECT_DOUBLE_EQ(J(0, 1), 0.0);
  EXPECT_DOUBLE_EQ(J(1, 1), 4.0);
}

/** DET Jacobian Physical Triangle
 * Test if the Determinant of the Jacobian matrix for the physical triangle
 * with nodes (1,2), (4,2), (1,6), (2.5, 2), (2.5, 4), (1, 4)
 * is 12
 */
TEST(FEMap2DTest, DetJacobianForPhysicalTriangle6Linear) {
  const P2_FE P2_element;

  const FEMap2D phys_triangle(phys_element_tr6, phys_nodes_tr6_linear,
                              P2_element);

  const auto detJ = phys_triangle.det_jacobian(RefNode{1.0, 0.0});

  EXPECT_DOUBLE_EQ(detJ, 12.0);
}

/** Jacobian Physical Triangle
 * NONLINEAR ISOPARAMETRIC MAPPING  [Triangle6] -> [P2]
 * Test if the Jacobian matrix for the physical triangle
 * with nodes (1,2), (4,2), (1,6), (2.5, 2), (3, 4.5), (1, 4)
 * is the matrix [[3+2*eta, 2*xi], [2*eta, 4+2*xi]]
 */
TEST(FEMap2DTest, JacobianForPhysicalTriangle6NonLinear) {
  const P2_FE P2_element;

  const FEMap2D phys_triangle(phys_element_tr6, phys_nodes_tr6_nonlinear,
                              P2_element);

  auto expected_J = [](const RefNode& ref_node) {
    Matrix2d J = Matrix2d::Zero();

    J(0, 0) = 3.0 + 2.0 * ref_node.eta;
    J(0, 1) = 2.0 * ref_node.xi;
    J(1, 0) = 2.0 * ref_node.eta;
    J(1, 1) = 4.0 + 2.0 * ref_node.xi;

    return J;
  };

  const auto ref_nodes = P2_element.get_ref_nodes();

  for (const auto& ref_node : ref_nodes) {
    const auto calc_J = phys_triangle.jacobian(ref_node);
    const auto exp_J = expected_J(ref_node);

    for (Eigen::Index i = 0; i < exp_J.rows(); ++i) {
      for (Eigen::Index j = 0; j < exp_J.cols(); ++j) {
        EXPECT_NEAR(calc_J(i, j), exp_J(i, j), test_dtol);
      }
    }
  }
}

/** DET Jacobian Physical Triangle
 * Test if the Determinant of the Jacobian matrix for the physical triangle
 * with nodes (1,2), (4,2), (1,6), (2.5, 2), (3, 4.5), (1, 4)
 * is 12 + 6*xi + 8*eta
 */
TEST(FEMap2DTest, DetJacobianForPhysicalTriangle6NonLinear) {
  const P2_FE P2_element;

  const FEMap2D phys_triangle(phys_element_tr6, phys_nodes_tr6_nonlinear,
                              P2_element);

  auto expected_detJ = [](const RefNode& ref_node) {
    return 12.0 + 6.0 * ref_node.xi + 8.0 * ref_node.eta;
  };

  const auto ref_nodes = P2_element.get_ref_nodes();

  for (const auto& ref_node : ref_nodes) {
    const auto calc_detJ = phys_triangle.det_jacobian(ref_node);
    const auto exp_detJ = expected_detJ(ref_node);

    EXPECT_NEAR(calc_detJ, exp_detJ, test_dtol);
  }
}

/** Inverse Jacobian Transpose Physical Triangle
 * Test if the inverse Jacobian matrix transposed for the physical triangle
 * with nodes (1,2), (4,2), (1,6)
 * is [[1/3,0],[0,1/4]]
 */
TEST(FEMap2DTest, InvJacobianTForPhysicalTriangle3) {
  const P1_FE P1_element;

  const FEMap2D phys_triangle(phys_element_tr3, phys_nodes_tr3, P1_element);

  const auto JinvT = phys_triangle.jacobianInvT(RefNode{1.0, 0.0});

  EXPECT_NEAR(JinvT(0, 0), 1.0 / 3.0, test_dtol);
  EXPECT_NEAR(JinvT(1, 0), 0.0, test_dtol);
  EXPECT_NEAR(JinvT(0, 1), 0.0, test_dtol);
  EXPECT_NEAR(JinvT(1, 1), 1.0 / 4.0, test_dtol);
}

/** Inverse Jacobian Transpose Physical Triangle
 * Test if the inverse Jacobian matrix transposed for the physical triangle
 * with nodes (1,2), (4,2), (1,6), (2.5, 2), (2.5, 4), (1, 4)
 * is [[1/3,0],[0,1/4]]
 */
TEST(FEMap2DTest, InvJacobianTForPhysicalTriangle6Linear) {
  const P2_FE P2_element;

  const FEMap2D phys_triangle(phys_element_tr6, phys_nodes_tr6_linear,
                              P2_element);

  const auto JinvT = phys_triangle.jacobianInvT(RefNode{1.0, 0.0});

  EXPECT_NEAR(JinvT(0, 0), 1.0 / 3.0, test_dtol);
  EXPECT_NEAR(JinvT(1, 0), 0.0, test_dtol);
  EXPECT_NEAR(JinvT(0, 1), 0.0, test_dtol);
  EXPECT_NEAR(JinvT(1, 1), 1.0 / 4.0, test_dtol);
}

/** Inverse Jacobian Transpose Physical Triangle
 * Test if the inverse Jacobian matrix transposed for the physical triangle
 * with nodes (1,2), (4,2), (1,6), (2.5, 2), (3, 4.5), (1, 4)
 * is 1/|12+6*xi+8*eta| * [[4 + 2*xi, -2*xi], [-2*eta, 3 + 2*eta]]
 */
TEST(FEMap2DTest, InvJacobianTForPhysicalTriangle6NonLinear) {
  const P2_FE P2_element;

  const FEMap2D phys_triangle(phys_element_tr6, phys_nodes_tr6_nonlinear,
                              P2_element);

  auto expected_JinvT = [](const RefNode& ref_node) {
    const double xi = ref_node.xi;
    const double eta = ref_node.eta;
    const double factor = 1.0 / std::abs(12.0 + 6.0 * xi + 8.0 * eta);

    Matrix2d JinvT = Matrix2d::Zero();

    JinvT(0, 0) = (4.0 + 2.0 * xi) * factor;
    JinvT(0, 1) = (-2.0 * eta) * factor;
    JinvT(1, 0) = (-2.0 * xi) * factor;
    JinvT(1, 1) = (3.0 + 2.0 * eta) * factor;

    return JinvT;
  };

  const std::vector<RefNode> ref_nodes = P2_element.get_ref_nodes();

  for (const RefNode& ref_node : ref_nodes) {
    const auto calculated_JinvT = phys_triangle.jacobianInvT(ref_node);
    const auto exp_JinvT = expected_JinvT(ref_node);

    for (Eigen::Index i = 0; i < calculated_JinvT.rows(); i++) {
      for (Eigen::Index j = 0; j < calculated_JinvT.cols(); j++) {
        EXPECT_NEAR(calculated_JinvT(i, j), exp_JinvT(i, j), test_dtol);
      }
    }
  }
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
                        .node_ids = {0, 1},
                        .edge_ids = {0}};

  const P1_FE P1_element;

  EXPECT_THROW(FEMap2D mapping(element, nodes, P1_element),
               std::invalid_argument);
}

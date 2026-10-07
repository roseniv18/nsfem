#include <gtest/gtest.h>

#include <Eigen/Sparse>
#include <memory>

#include "FEM/assemble/assemble.h"
#include "FEM/finite_element/dof_handler.h"
#include "FEM/finite_element/finite_element.h"
#include "helpers/helpers.h"
#include "mesh/parser.h"

using SparseMatrix = Eigen::SparseMatrix<double>;

/**
 * Global stiffness matrix assembly test for simple mesh.
 *
 * Mesh consists of 2 triangles with nodes
 * (0,0), (1,0), (1,1), (0,1)
 */
TEST(GlobalAssemblyTest, SimpleMesh) {
  Mesh mesh;

  mesh.nodes = {
      {1, 0.0, 0.0},
      {2, 1.0, 0.0},
      {3, 1.0, 1.0},
      {4, 0.0, 1.0},
  };

  mesh.elements = {
      Element{
          .dim = 2,
          .element_tag = 1,
          .type = ElementType::Triangle3,
          .physical_tags = {},
          .node_ids = {0, 1, 2},
      },
      Element{
          .dim = 2,
          .element_tag = 2,
          .type = ElementType::Triangle3,
          .physical_tags = {},
          .node_ids = {0, 2, 3},
      },
  };

  const std::unique_ptr<FE> fe = FE::build_fe_type(FEType::P1);
  const DOFHandler dofh(mesh, *fe);

  const SparseMatrix K = asm_global_stiffness_matr(mesh, *fe, dofh);

  const double expected[4][4] = {
      {1.0, -0.5, 0.0, -0.5},
      {-0.5, 1.0, -0.5, 0.0},
      {0.0, -0.5, 1.0, -0.5},
      {-0.5, 0.0, -0.5, 1.0},
  };

  ASSERT_EQ(K.rows(), 4);
  ASSERT_EQ(K.cols(), 4);

  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      EXPECT_NEAR(K.coeff(i, j), expected[i][j], test_dtol);
    }
  }
}

/**
 * Global stiffness matrix symmetry test.
 */
TEST(GlobalAssemblyTest, IsGlobalStiffnessMatrixSymmetric) {
  Mesh mesh;

  mesh.nodes = {
      {1, 0.0, 0.0},
      {2, 1.0, 0.0},
      {3, 1.0, 1.0},
      {4, 0.0, 1.0},
  };

  mesh.elements = {
      Element{
          .dim = 2,
          .element_tag = 1,
          .type = ElementType::Triangle3,
          .physical_tags = {},
          .node_ids = {0, 1, 2},
      },
      Element{
          .dim = 2,
          .element_tag = 2,
          .type = ElementType::Triangle3,
          .physical_tags = {},
          .node_ids = {0, 2, 3},
      },
  };

  const std::unique_ptr<FE> fe = FE::build_fe_type(FEType::P1);
  const DOFHandler dofh(mesh, *fe);

  const SparseMatrix K = asm_global_stiffness_matr(mesh, *fe, dofh);

  for (Eigen::Index i = 0; i < K.rows(); ++i) {
    for (Eigen::Index j = 0; j < K.cols(); ++j) {
      EXPECT_NEAR(K.coeff(i, j), K.coeff(j, i), test_dtol);
    }
  }
}

TEST(GlobalAssemblyTest, P2StiffnessMatrixSize) {
  Mesh mesh = make_two_triangle_mesh();

  const auto fe = FE::build_fe_type(FEType::P2);
  const DOFHandler dofh(mesh, *fe);

  const SparseMatrix K = asm_global_stiffness_matr(mesh, *fe, dofh);

  EXPECT_EQ(K.rows(), 9);
  EXPECT_EQ(K.cols(), 9);
}

TEST(GlobalAssemblyTest, P2StiffnessIsSymmetric) {
  Mesh mesh = make_two_triangle_mesh();

  const auto fe = FE::build_fe_type(FEType::P2);
  const DOFHandler dofh(mesh, *fe);

  const SparseMatrix K = asm_global_stiffness_matr(mesh, *fe, dofh);

  for (Eigen::Index i = 0; i < K.rows(); ++i) {
    for (Eigen::Index j = 0; j < K.cols(); ++j) {
      EXPECT_NEAR(K.coeff(i, j), K.coeff(j, i), test_dtol);
    }
  }
}

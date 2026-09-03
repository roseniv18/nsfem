#include <gtest/gtest.h>
#include "assemble/assemble.h"
#include "linalg/matrix.h"
#include "mesh/parser.h"

/** Global assembly test for simple mesh
 * Mesh consists of 2 triangles with nodes
 * (0,0), (1,0), (1,1)
 */
TEST(GlobalAssemblyTest, SimpleMesh) {
  Mesh mesh;

  mesh.nodes = {{1, 0.0, 0.0}, {2, 1.0, 0.0}, {3, 1.0, 1.0}, {4, 0.0, 1.0}};

  mesh.elements = {
      Element{.dim = 2,
              .element_tag = 1,
              .type = ElementType::Triangle3,
              .physical_tags = {},
              .node_indices = {0, 1, 2}},
      Element{.dim = 2,
              .element_tag = 2,
              .type = ElementType::Triangle3,
              .physical_tags = {},
              .node_indices = {0, 2, 3}},
  };

  const Matrix<double> K = assemble_gs_matrix(mesh);

  const double expected[4][4] = {{1.0, -0.5, 0.0, -0.5},
                                 {-0.5, 1.0, -0.5, 0.0},
                                 {0.0, -0.5, 1.0, -0.5},
                                 {-0.5, 0.0, -0.5, 1.0}};

  for (std::size_t i = 0; i < 4; i++) {
    for (std::size_t j = 0; j < 4; j++) {
      EXPECT_NEAR(K(i, j), expected[i][j], 1e-12);
    }
  }
}

/** Global stiffness matrix symmetry test
 */
TEST(GlobalAssemblyTest, IsGlobalStiffnessMatrixSymmetric) {
  Mesh mesh;

  mesh.nodes = {{1, 0.0, 0.0}, {2, 1.0, 0.0}, {3, 1.0, 1.0}, {4, 0.0, 1.0}};

  mesh.elements = {
      Element{.dim = 2,
              .element_tag = 1,
              .type = ElementType::Triangle3,
              .physical_tags = {},
              .node_indices = {0, 1, 2}},
      Element{.dim = 2,
              .element_tag = 2,
              .type = ElementType::Triangle3,
              .physical_tags = {},
              .node_indices = {0, 2, 3}},
  };

  const Matrix<double> K = assemble_gs_matrix(mesh);

  for (std::size_t i = 0; i < 4; i++) {
    for (std::size_t j = 0; j < 4; j++) {
      EXPECT_NEAR(K(i, j), K(j, i), 1e-12);
    }
  }
}

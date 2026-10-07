#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include "FEM/finite_element/finite_element.h"
#include "helpers/helpers.h"
#include "mesh/mesh.h"

/**
 * Test DOFHandler on a mesh consisting of 2 triangles with a shared edge:
 * Nodes:
 * 0 = (0, 0)
 * 1 = (1, 0)
 * 2 = (0, 1)
 * 3 = (1, 1)
 *
 * Triangle 1 = (0, 1, 2)
 * Triangle 2 = (1, 3, 2)
 *
 * This gives:
 * 4 vertices
 * 5 unique edges
 * For P1: 4 DOFs
 * For P2: 9 DOFs
 */

//  P1 DOFs count
TEST(DOFHandlerTest, P1GlobalDOFCount) {
  Mesh mesh = make_two_triangle_mesh();

  const std::unique_ptr<FE> fe = FE::build_fe_type(FEType::P1);
  const DOFHandler dofh(mesh, *fe);

  EXPECT_EQ(dofh.get_global_ndofs(), 4);
}

//  P2 DOFs count
TEST(DOFHandlerTest, P2GlobalDOFCount) {
  Mesh mesh = make_two_triangle_mesh();

  const std::unique_ptr<FE> fe = FE::build_fe_type(FEType::P2);
  const DOFHandler dofh(mesh, *fe);

  EXPECT_EQ(dofh.get_global_ndofs(), 9);
}

// P1 DOF Indices
TEST(DOFHandlerTest, P1ElementDOFIndices) {
  Mesh mesh = make_two_triangle_mesh();

  const std::unique_ptr<FE> fe = FE::build_fe_type(FEType::P2);
  const DOFHandler dofh(mesh, *fe);

  const auto el_0_dof_indices = dofh.get_element_dof_indices(0);
  const auto el_1_dof_indices = dofh.get_element_dof_indices(1);

  const std::vector<int> el_0_expected_indices{0, 1, 2};
  const std::vector<int> el_1_expected_indices{1, 2, 3};

  for (std::size_t i = 0; i < 3; i++) {
    EXPECT_EQ(el_0_dof_indices.at(i), el_0_expected_indices.at(i));
    EXPECT_EQ(el_1_dof_indices.at(i), el_1_expected_indices.at(i));
  }
}

// P2 DOF Indices
TEST(DOFHandlerTest, P2ElementDOFIndices) {
  Mesh mesh = make_two_triangle_mesh();

  const std::unique_ptr<FE> fe = FE::build_fe_type(FEType::P2);
  const DOFHandler dofh(mesh, *fe);

  const auto el_0_dof_indices = dofh.get_element_dof_indices(0);
  const auto el_1_dof_indices = dofh.get_element_dof_indices(1);

  const std::vector<int> el_0_expected_indices{
      0, 1, 2,  // vertices
      4, 5, 6   // edges
  };

  const std::vector<int> el_1_expected_indices{
      1, 2, 3,  // vertices
      5, 7, 8   // edges
  };

  for (std::size_t i = 0; i < 3; i++) {
    EXPECT_EQ(el_0_dof_indices.at(i), el_0_expected_indices.at(i));
    EXPECT_EQ(el_1_dof_indices.at(i), el_1_expected_indices.at(i));
  }
}

// P2 shared edge must have the same global DOF
TEST(DOFHandlerTest, P2SharedEdgeHasSameGlobalDOF) {
  Mesh mesh = make_two_triangle_mesh();

  const std::unique_ptr<FE> fe = FE::build_fe_type(FEType::P2);
  const DOFHandler dofh(mesh, *fe);

  const auto el_0_dof_indices = dofh.get_element_dof_indices(0);
  const auto el_1_dof_indices = dofh.get_element_dof_indices(1);

  // Element 0 = (0, 1, 2)
  // Shared edge (1, 2) -> local edge DOF 4
  const int el_0_shared_edge_dof = el_0_dof_indices.at(4);

  // Element 1 = (1, 2, 3)
  // Shared edge (1, 2) -> local edge DOF 3
  const int el_1_shared_edge_dof = el_1_dof_indices.at(3);

  EXPECT_EQ(el_0_shared_edge_dof, el_1_shared_edge_dof);
}

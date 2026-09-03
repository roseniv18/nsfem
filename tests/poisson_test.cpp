#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <numbers>
#include <vector>

#include "assemble/assemble.h"
#include "convergence/convergence.h"
#include "linalg/conjugate_gradient.h"
#include "mesh/parser.h"

namespace {

Mesh make_unit_square_mesh(int n) {
  Mesh mesh{};

  /*
      Unit square divided into n x n cells.

      Each square cell is split into two triangles:

      (i,j+1) ---- (i+1,j+1)
         |       / |
         |     /   |
         |   /     |
         | /       |
      (i,j) ---- (i+1,j)

      All triangles are oriented counter-clockwise,
      so det(J) > 0.
  */

  // ------------------------------------------------------------
  // Nodes
  // ------------------------------------------------------------

  const int nodes_per_row = n + 1;

  mesh.nodes.reserve(nodes_per_row * nodes_per_row);

  for (int j = 0; j <= n; ++j) {
    for (int i = 0; i <= n; ++i) {
      const double x = static_cast<double>(i) / static_cast<double>(n);

      const double y = static_cast<double>(j) / static_cast<double>(n);

      const int tag = j * nodes_per_row + i + 1;

      mesh.nodes.push_back(Node{tag, x, y, 0.0});
    }
  }

  // ------------------------------------------------------------
  // Physical group used for Dirichlet boundary
  // ------------------------------------------------------------

  mesh.physical_groups[1] = PhysicalGroup{.dim = 1, .name = "Dirichlet"};

  // ------------------------------------------------------------
  // Helper for converting (i,j) -> node index
  // ------------------------------------------------------------

  auto node_index = [nodes_per_row](int i, int j) {
    return static_cast<std::size_t>(j * nodes_per_row + i);
  };

  // ------------------------------------------------------------
  // Triangle elements
  // ------------------------------------------------------------

  int element_tag = 1;

  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const std::size_t bl = node_index(i, j);
      const std::size_t br = node_index(i + 1, j);
      const std::size_t tr = node_index(i + 1, j + 1);
      const std::size_t tl = node_index(i, j + 1);

      // Lower-right triangle
      mesh.elements.push_back(Element{.dim = 2,
                                      .element_tag = element_tag++,
                                      .type = ElementType::Triangle3,
                                      .physical_tags = {},
                                      .node_indices = {bl, br, tr}});

      // Upper-left triangle
      mesh.elements.push_back(Element{.dim = 2,
                                      .element_tag = element_tag++,
                                      .type = ElementType::Triangle3,
                                      .physical_tags = {},
                                      .node_indices = {bl, tr, tl}});
    }
  }

  // ------------------------------------------------------------
  // Boundary Line2 elements
  // ------------------------------------------------------------

  auto add_boundary_edge = [&](std::size_t a, std::size_t b) {
    mesh.elements.push_back(Element{.dim = 1,
                                    .element_tag = element_tag++,
                                    .type = ElementType::Line2,
                                    .physical_tags = {1},
                                    .node_indices = {a, b}});
  };

  // Bottom
  for (int i = 0; i < n; ++i) {
    add_boundary_edge(node_index(i, 0), node_index(i + 1, 0));
  }

  // Right
  for (int j = 0; j < n; ++j) {
    add_boundary_edge(node_index(n, j), node_index(n, j + 1));
  }

  // Top
  for (int i = 0; i < n; ++i) {
    add_boundary_edge(node_index(i, n), node_index(i + 1, n));
  }

  // Left
  for (int j = 0; j < n; ++j) {
    add_boundary_edge(node_index(0, j), node_index(0, j + 1));
  }

  return mesh;
}

}  // namespace

TEST(PoissonTest, L2Convergence) {
  const std::vector<int> resolutions{4, 8, 16, 32};

  std::vector<double> errors;

  for (const int n : resolutions) {
    Mesh mesh = make_unit_square_mesh(n);

    // Assemble system
    Matrix<double> K = assemble_gs_matrix(mesh);

    std::vector<double> rhs = assemble_gl_vector(mesh, func);

    // Find Dirichlet nodes
    const auto dirichlet_nodes = get_dirichlet_nodes(mesh);

    // Compute Dirichlet values
    const auto dirichlet_values =
        get_dirichlet_values(mesh, dirichlet_nodes, dir_func);

    // Apply BCs
    apply_dirichlet_bc(K, rhs, dirichlet_values);

    // Solve K u = f
    ConjugateGradient cg(K, rhs);

    const std::vector<double> solution = cg.solve();

    // Compute L2 error
    const double error = global_l2_err(mesh, solution);

    errors.push_back(error);

    std::cout << "n = " << n << ", L2 error = " << error << '\n';

    EXPECT_GT(error, 0.0);
  }

  // ------------------------------------------------------------
  // Check convergence rate
  // ------------------------------------------------------------

  for (std::size_t i = 1; i < errors.size(); ++i) {
    const double rate = std::log(errors[i - 1] / errors[i]) / std::log(2.0);

    EXPECT_GT(rate, 1.8);
    EXPECT_LT(rate, 2.2);
  }
}

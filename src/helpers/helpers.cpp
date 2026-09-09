#include "helpers/helpers.h"

std::string strip_quotes(std::string& str) {
  if (str.size() >= 2 && str.front() == '"' && str.back() == '"') {
    return str.substr(1, str.size() - 2);
  }

  return str;
}

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

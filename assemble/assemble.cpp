#include "assemble.h"
#include <stdlib.h>
#include "geometry/affine.h"

local_matr generate_ls_matrix(const Mesh& mesh, const Element& element) {
  local_matr ls_matrix;
  AffineMap am = compute_affine(element);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      double dot = (am.phys_grads[i].x * am.phys_grads[j].x) +
                   (am.phys_grads[i].y * am.phys_grads[j].y);
      ls_matrix[i][j] = 0.5 * std::abs(am.detJ) * dot;
    }
  }

  return ls_matrix;
}

std::vector<std::vector<double>> assemble_gs_matrix(const Mesh& mesh) {
  const std::size_t n = mesh.nodes.size();

  std::vector<std::vector<double>> gs_matrix(n, std::vector<double>(n, 0.0));

  for (const Element& element : mesh.elements) {
    if (element.type == 2) {
      local_matr ls_matrix = generate_ls_matrix(mesh, element);

      for (std::size_t i = 0; i < element.node_indices.size(); ++i) {
        for (std::size_t j = 0; j < element.node_indices.size(); ++j) {
          const std::size_t I = element.node_indices[i];
          const std::size_t J = element.node_indices[j];

          gs_matrix[I][J] += ls_matrix[i][j];
        }
      }
    }
  }

  return gs_matrix;
}

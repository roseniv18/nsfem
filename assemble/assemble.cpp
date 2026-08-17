#include "assemble.h"
#include <stdlib.h>
#include "geometry/affine.h"

// generete local stiffness matrix
local_matr generate_ls_matrix(const Element& element) {
  local_matr ls_matrix{};
  AffineMap am = compute_affine(element);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      /** this calculation assumes:
       * 	1. affine mapping
       *  	2. constant gradients (true for linear Lagrange triangles)
       */
      double dot = (am.phys_grads[i].x * am.phys_grads[j].x) +
                   (am.phys_grads[i].y * am.phys_grads[j].y);
      ls_matrix[i][j] = 0.5 * std::abs(am.detJ) * dot;
    }
  }

  return ls_matrix;
}

// assemble global stiffness matrix
std::vector<std::vector<double>> assemble_gs_matrix(const Mesh& mesh) {
  const std::size_t n = mesh.nodes.size();

  std::vector<std::vector<double>> gs_matrix(n, std::vector<double>(n, 0.0));

  for (const Element& element : mesh.elements) {
    if (element.type == 2) {
      local_matr ls_matrix = generate_ls_matrix(element);

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

// generate local load vector
local_vec generate_loc_vector(const Element& element,
                              double (*f)(const Point2D&)) {
  local_vec lv{};
  AffineMap am = compute_affine(element);

  auto bfs = bfs_at_quad();

  for (std::size_t q = 0; q < quad_nodes.size(); q++) {
    Point2D ref = quad_nodes[q];
    Point2D phys = map_to_phys(element, ref);

    double f_val = f(phys);

    for (int i = 0; i < 3; i++) {
      lv[i] += std::abs(am.detJ) * quad_weights[q] * bfs[i][q] * f_val;
    }
  }

  return lv;
}

// assemble global load vector
std::vector<double> assemble_gl_vector(const Mesh& mesh,
                                       double (*f)(const Point2D&)) {
  const std::size_t n = mesh.nodes.size();

  std::vector<double> gl_vector(n, 0.0);

  for (const Element& element : mesh.elements) {
    if (element.type == 2) {
      local_vec lv = generate_loc_vector(element, f);

      for (std::size_t i = 0; i < element.node_indices.size(); ++i) {
        const std::size_t I = element.node_indices[i];

        gl_vector.at(I) += lv.at(i);
      }
    }
  }

  return gl_vector;
}

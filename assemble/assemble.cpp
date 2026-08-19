#include "assemble.h"
#include <stdlib.h>
#include "geometry/affine.h"

// generate local stiffness matrix
Matrix<double> generate_ls_matrix(const Element& element, const Mesh& mesh) {
  Matrix<double> ls_matrix(3, 3);
  const auto el_nodes = get_element_nodes(element, mesh);
  AffineMap am = compute_affine(element, el_nodes);

  for (int i = 0; i < ls_matrix.n; i++) {
    for (int j = 0; j < ls_matrix.m; j++) {
      /** this calculation assumes:
       * 	1. affine mapping
       *  	2. constant gradients (true for linear Lagrange triangles)
       */
      double dot = (am.phys_grads[i].x * am.phys_grads[j].x) +
                   (am.phys_grads[i].y * am.phys_grads[j].y);
      ls_matrix(i, j) = 0.5 * std::abs(am.detJ) * dot;
    }
  }

  return ls_matrix;
}

// assemble global stiffness matrix
Matrix<double> assemble_gs_matrix(const Mesh& mesh) {
  const std::size_t n = mesh.nodes.size();

  Matrix<double> gs_matrix(n, n);

  for (const Element& element : mesh.elements) {
    if (element.type == 2) {
      Matrix<double> ls_matrix = generate_ls_matrix(element, mesh);

      for (std::size_t i = 0; i < element.node_indices.size(); ++i) {
        for (std::size_t j = 0; j < element.node_indices.size(); ++j) {
          const std::size_t I = element.node_indices[i];
          const std::size_t J = element.node_indices[j];

          gs_matrix(I, J) += ls_matrix(i, j);
        }
      }
    }
  }

  return gs_matrix;
}

// generate local load vector
local_vec generate_loc_vector(const Element& element,
                              double (*f)(const Point2D&),
                              const Mesh& mesh) {
  local_vec lv{};
  const auto el_nodes = get_element_nodes(element, mesh);
  AffineMap am = compute_affine(element, el_nodes);

  auto bfs = bfs_at_quad();

  for (std::size_t q = 0; q < quad_nodes.size(); q++) {
    Point2D ref = quad_nodes[q];
    Point2D phys = map_to_phys(element, ref, el_nodes);

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
      local_vec lv = generate_loc_vector(element, f, mesh);

      for (std::size_t i = 0; i < element.node_indices.size(); ++i) {
        const std::size_t I = element.node_indices[i];

        gl_vector.at(I) += lv.at(i);
      }
    }
  }

  return gl_vector;
}

// apply Dirichlet boundary conditions
void apply_dirichlet_bc(Matrix<double>& K,
                        std::vector<double>& f,
                        const std::unordered_map<int, double>& dirichlet_vals) {
  for (const auto& [i, val] : dirichlet_vals) {
    // modify RHS
    for (int j = 0; j < K.n; j++) {
      if (j != i) {
        f[j] -= K(j, i) * val;
      }
    }

    // zero out row
    for (int j = 0; j < K.m; j++) {
      K(i, j) = 0;
    }

    // zero out column
    for (int j = 0; j < K.n; j++) {
      K(j, i) = 0;
    }

    K(i, i) = 1.0;
    f.at(i) = val;
  }
}

#include "assemble.h"

// generate local stiffness matrix
Matrix<double> gen_local_stiffness_matr(const TriangleGEO& element) {
  Matrix<double> ls_matrix(3, 3);

  auto phys_grads = element.get_phys_grads();
  const auto detJ = element.det_jacobian();

  for (int i = 0; i < ls_matrix.n; i++) {
    for (int j = 0; j < ls_matrix.m; j++) {
      /** this calculation assumes:
       * 	1. affine mapping
       *  	2. constant gradients (true for linear Lagrange triangles)
       */
      double dot = (phys_grads.at(i).x * phys_grads.at(j).x) +
                   (phys_grads.at(i).y * phys_grads.at(j).y);
      ls_matrix(i, j) = 0.5 * std::abs(detJ) * dot;
    }
  }

  return ls_matrix;
}

// generate local mass matrix
Matrix<double> gen_local_mass_matr(const TriangleGEO& element) {
  Matrix<double> lm_matrix(3, 3);

  const auto detJ = element.det_jacobian();

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (i == j) {
        lm_matrix(i, j) = 0.5 * detJ / 6.0;
      } else {
        lm_matrix(i, j) = 0.5 * detJ / 12.0;
      }
    }
  }

  return lm_matrix;
}

// assemble global mass matrix
Matrix<double> asm_global_mass_matr(const Mesh& mesh) {
  const std::size_t n = mesh.nodes.size();

  Matrix<double> gm_matrix(n, n);

  for (const Element& element : mesh.elements) {
    if (element.type == ElementType::Triangle3) {
      const auto el_nodes = get_element_nodes(element, mesh);
      const TriangleGEO tr_element(element, el_nodes);

      Matrix<double> ls_matrix = gen_local_stiffness_matr(tr_element);

      for (std::size_t i = 0; i < element.node_indices.size(); ++i) {
        for (std::size_t j = 0; j < element.node_indices.size(); ++j) {
          const std::size_t I = element.node_indices[i];
          const std::size_t J = element.node_indices[j];

          gm_matrix(I, J) += ls_matrix(i, j);
        }
      }
    }
  }

  return gm_matrix;
}

// assemble global stiffness matrix
Matrix<double> asm_global_stiffness_matr(const Mesh& mesh) {
  const std::size_t n = mesh.nodes.size();

  Matrix<double> gs_matrix(n, n);

  for (const Element& element : mesh.elements) {
    if (element.type == ElementType::Triangle3) {
      const auto el_nodes = get_element_nodes(element, mesh);
      const TriangleGEO tr_element(element, el_nodes);

      Matrix<double> ls_matrix = gen_local_stiffness_matr(tr_element);

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
local_vec gen_local_vec(const TriangleGEO& element, STFunction func, double t) {
  local_vec lv{};

  auto phys_points = element.get_phys_coords();
  auto detJ = element.det_jacobian();
  auto bfs = bfs_at_quad();

  for (std::size_t q = 0; q < quad_nodes.size(); q++) {
    double f_val = func(phys_points[q].x, phys_points[q].y, t);

    for (int i = 0; i < 3; i++) {
      lv[i] += std::abs(detJ) * quad_weights[q] * bfs[i][q] * f_val;
    }
  }

  return lv;
}

// assemble global load vector
std::vector<double> asm_global_vec(const Mesh& mesh,
                                   STFunction func,
                                   double t) {
  const std::size_t n = mesh.nodes.size();

  std::vector<double> gl_vector(n, 0.0);

  for (const Element& element : mesh.elements) {
    if (element.type == ElementType::Triangle3) {
      const auto el_nodes = get_element_nodes(element, mesh);
      const TriangleGEO tr_element(element, el_nodes);

      local_vec lv = gen_local_vec(tr_element, func, t);

      for (std::size_t i = 0; i < element.node_indices.size(); ++i) {
        const std::size_t I = element.node_indices[i];

        gl_vector.at(I) += lv.at(i);
      }
    }
  }

  return gl_vector;
}

// apply Dirichlet boundary conditions
void apply_dirichlet_bc(Matrix<double>& A,
                        std::vector<double>& vec,
                        const std::vector<bool>& is_dirichlet,
                        const std::vector<double>& dirichlet_vals) {
  for (int i = 0; i < A.n; i++) {
    if (is_dirichlet.at(i)) {
      const double val = dirichlet_vals.at(i);

      // modify RHS
      for (int j = 0; j < A.n; j++) {
        if (j != i) {
          vec[j] -= A(j, i) * val;
        }
      }

      // zero out row
      for (int j = 0; j < A.n; j++) {
        A(i, j) = 0;
      }

      // zero out column
      for (int j = 0; j < A.m; j++) {
        A(j, i) = 0;
      }

      A(i, i) = 1.0;
      vec.at(i) = val;
    }
  }
}

void apply_dirichlet_bc_matr(Matrix<double>& A,
                             const std::vector<bool>& is_dirichlet) {
  for (int i = 0; i < A.n; i++) {
    if (is_dirichlet.at(i)) {
      // zero out row
      for (int j = 0; j < A.n; j++) {
        A(i, j) = 0;
      }

      // zero out column
      for (int j = 0; j < A.m; j++) {
        A(j, i) = 0;
      }

      A(i, i) = 1.0;
    }
  }
}

// apply Dirichlet boundary conditions
void apply_dirichlet_bc_vec(Matrix<double>& A,
                            std::vector<double>& vec,
                            const std::vector<double>& dirichlet_vals,
                            const std::vector<bool>& is_dirichlet) {
  for (int i = 0; i < A.n; i++) {
    const double val = dirichlet_vals.at(i);

    if (is_dirichlet.at(i)) {
      // modify RHS
      for (int j = 0; j < A.n; j++) {
        if (j != i) {
          vec[j] -= A(j, i) * val;
        }
      }

      vec.at(i) = val;
    }
  }
}

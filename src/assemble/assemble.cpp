#include "assemble.h"

// generate local stiffness matrix
MatrixXd gen_local_stiffness_matr(const TriangleGEO& element) {
  MatrixXd ls_matrix = MatrixXd::Zero(3, 3);

  auto phys_grads = element.get_phys_grads();
  const auto detJ = element.det_jacobian();

  for (int i = 0; i < ls_matrix.cols(); i++) {
    for (int j = 0; j < ls_matrix.rows(); j++) {
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
MatrixXd gen_local_mass_matr(const TriangleGEO& element) {
  MatrixXd lm_matrix = MatrixXd::Zero(3, 3);

  const auto detJ = element.det_jacobian();

  for (int i = 0; i < lm_matrix.cols(); i++) {
    for (int j = 0; j < lm_matrix.rows(); j++) {
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
SparseMatrix asm_global_mass_matr(const Mesh& mesh) {
  const std::size_t n = mesh.nodes.size();

  std::vector<Eigen::Triplet<double>> triplets;

  //   Each triangle contributes at most 3*3=9 entries (linear P1)
  triplets.reserve(mesh.elements.size() * 9);

  for (const Element& element : mesh.elements) {
    if (element.type == ElementType::Triangle3) {
      const auto el_nodes = get_element_nodes(element, mesh);
      const TriangleGEO tr_element(element, el_nodes);

      const MatrixXd local_mass_matrix = gen_local_mass_matr(tr_element);

      for (std::size_t i = 0; i < element.node_indices.size(); i++) {
        for (std::size_t j = 0; j < element.node_indices.size(); j++) {
          const std::size_t I = element.node_indices.at(i);
          const std::size_t J = element.node_indices.at(j);

          triplets.emplace_back(static_cast<Eigen::Index>(I),
                                static_cast<Eigen::Index>(J),
                                local_mass_matrix(i, j));
        }
      }
    }
  }

  SparseMatrix global_mass_matrix(static_cast<Eigen::Index>(n),
                                  static_cast<Eigen::Index>(n));

  // setFromTriplets handles duplicate entries by summing them
  // this is exactly what FEM requires
  global_mass_matrix.setFromTriplets(triplets.begin(), triplets.end());

  // store in compressed memory format
  global_mass_matrix.makeCompressed();

  return global_mass_matrix;
}

// assemble global stiffness matrix
SparseMatrix asm_global_stiffness_matr(const Mesh& mesh) {
  const std::size_t n = mesh.nodes.size();

  std::vector<Eigen::Triplet<double>> triplets;

  //   Each triangle contributes at most 3*3=9 entries (linear P1)
  triplets.reserve(mesh.elements.size() * 9);

  for (const Element& element : mesh.elements) {
    if (element.type == ElementType::Triangle3) {
      const auto el_nodes = get_element_nodes(element, mesh);
      const TriangleGEO tr_element(element, el_nodes);

      const MatrixXd local_stiffness_matrix =
          gen_local_stiffness_matr(tr_element);

      for (std::size_t i = 0; i < element.node_indices.size(); i++) {
        for (std::size_t j = 0; j < element.node_indices.size(); j++) {
          const std::size_t I = element.node_indices.at(i);
          const std::size_t J = element.node_indices.at(j);

          triplets.emplace_back(static_cast<Eigen::Index>(I),
                                static_cast<Eigen::Index>(J),
                                local_stiffness_matrix(i, j));
        }
      }
    }
  }

  SparseMatrix global_stiffness_matrix(static_cast<Eigen::Index>(n),
                                       static_cast<Eigen::Index>(n));

  // setFromTriplets handles duplicate entries by summing them
  // this is exactly what FEM requires
  global_stiffness_matrix.setFromTriplets(triplets.begin(), triplets.end());

  // store in compressed memory format
  global_stiffness_matrix.makeCompressed();

  return global_stiffness_matrix;
}

// generate local load vector
VectorXd gen_local_vec(const TriangleGEO& element, STFunction func, double t) {
  VectorXd lv = VectorXd::Zero(3);

  auto phys_points = element.get_phys_coords();
  auto detJ = element.det_jacobian();
  auto bfs = bfs_at_quad();

  for (std::size_t q = 0; q < quad_nodes.size(); q++) {
    double f_val = func(phys_points[q].x, phys_points[q].y, t);

    for (int i = 0; i < 3; i++) {
      lv(i) += std::abs(detJ) * quad_weights[q] * bfs[i][q] * f_val;
    }
  }

  return lv;
}

// assemble global load vector
VectorXd asm_global_vec(const Mesh& mesh, STFunction func, double t) {
  const std::size_t n = mesh.nodes.size();

  VectorXd gl_vector = VectorXd::Zero(n);

  for (const Element& element : mesh.elements) {
    if (element.type == ElementType::Triangle3) {
      const auto el_nodes = get_element_nodes(element, mesh);
      const TriangleGEO tr_element(element, el_nodes);

      VectorXd lv = gen_local_vec(tr_element, func, t);

      for (std::size_t i = 0; i < element.node_indices.size(); ++i) {
        const std::size_t I = element.node_indices[i];

        gl_vector(I) += lv(i);
      }
    }
  }

  return gl_vector;
}

// apply Dirichlet boundary conditions
void apply_dirichlet_bc(SparseMatrix& A,
                        VectorXd& vec,
                        const std::vector<bool>& is_dirichlet,
                        const std::vector<double>& dirichlet_vals) {
  // RHS is modified using the original matrix
  apply_dirichlet_bc_vec(A, vec, is_dirichlet, dirichlet_vals);

  // Apply BC to matrix
  apply_dirichlet_bc_matr(A, is_dirichlet);
}

void apply_dirichlet_bc_matr(SparseMatrix& A,
                             const std::vector<bool>& is_dirichlet) {
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(A.nonZeros());

  for (int k = 0; k < A.outerSize(); k++) {
    for (SparseMatrix::InnerIterator it; it; ++it) {
      const Eigen::Index row = it.row();
      const Eigen::Index col = it.col();

      // Copy the original value unchanged, if the row and col are not Dirichlet
      // otherwise, leave row and col value of 0
      if (!is_dirichlet.at(row) && !is_dirichlet.at(col)) {
        triplets.emplace_back(row, col, it.value());
      }
    }
  }

  //   Set diagonal to 1
  for (Eigen::Index i = 0; i < A.rows(); i++) {
    if (is_dirichlet.at(i)) {
      triplets.emplace_back(i, i, 1.0);
    }
  }

  SparseMatrix A_new(A.rows(), A.cols());

  A_new.setFromTriplets(triplets.begin(), triplets.end());

  A = std::move(A_new);
}

// apply Dirichlet boundary conditions
void apply_dirichlet_bc_vec(const SparseMatrix& A,
                            VectorXd& vec,
                            const std::vector<bool>& is_dirichlet,
                            const std::vector<double>& dirichlet_vals) {
  // Default SparseMatrix storage is column-major, so loop through columns
  for (int k = 0; k < A.outerSize(); k++) {
    for (SparseMatrix::InnerIterator it(A, k); it; ++it) {
      const Eigen::Index row = it.row();
      const Eigen::Index col = it.col();

      //   b_row -= A_col,row
      if (is_dirichlet.at(col) && row != col) {
        vec(row) -= it.value() * dirichlet_vals.at(col);
      }
    }
  }

  //   apply Dirichlet value
  for (Eigen::Index i = 0; i < A.rows(); i++) {
    if (is_dirichlet.at(i)) {
      vec(i) = dirichlet_vals.at(i);
    }
  }
}

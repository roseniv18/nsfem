#include "assemble.h"

// generate local stiffness matrix
MatrixXd gen_local_stiffness_matr(const FEMap2D& mapping, const FE& element) {
  const int ndofs = element.get_ndofs();

  MatrixXd ls_matrix = MatrixXd::Zero(ndofs, ndofs);

  const double detJ = std::abs(mapping.det_jacobian());
  const MatrixXd JinvT = mapping.jacobianInvT();

  for (std::size_t q = 0; q < quad_nodes.size(); q++) {
    const double xi = quad_nodes.at(q).at(0);
    const double eta = quad_nodes.at(q).at(1);

    const auto ref_grads = element.evaluate_grad_bfs(xi, eta);

    // Physical gradients' values at this quadrature point (xi, eta)
    std::vector<Point2D> phys_grads(ndofs);

    for (int i = 0; i < ndofs; i++) {
      phys_grads.at(i).x = JinvT(0, 0) * ref_grads.at(i).at(0) +
                           JinvT(0, 1) * ref_grads.at(i).at(1);

      phys_grads.at(i).y = JinvT(1, 0) * ref_grads.at(i).at(0) +
                           JinvT(1, 1) * ref_grads.at(i).at(1);
    }

    for (int i = 0; i < ndofs; i++) {
      for (int j = 0; j < ndofs; j++) {
        const double dot = phys_grads.at(i).x * phys_grads.at(j).x +
                           phys_grads.at(i).y * phys_grads.at(j).y;

        ls_matrix(i, j) += detJ * quad_weights.at(q) * dot;
      }
    }
  }

  return ls_matrix;
}

// generate local mass matrix
MatrixXd gen_local_mass_matr(const FEMap2D& mapping, const FE& element) {
  const int ndofs = element.get_ndofs();

  MatrixXd lm_matrix = MatrixXd::Zero(ndofs, ndofs);

  const double detJ = std::abs(mapping.det_jacobian());

  for (std::size_t q = 0; q < quad_nodes.size(); q++) {
    const double xi = quad_nodes.at(q).at(0);
    const double eta = quad_nodes.at(q).at(1);

    const auto bfs = element.evaluate_bfs(xi, eta);

    for (int i = 0; i < ndofs; i++) {
      for (int j = 0; j < ndofs; j++) {
        lm_matrix(i, j) += detJ * quad_weights.at(q) * bfs.at(i) * bfs.at(j);
      }
    }
  }

  return lm_matrix;
}

// assemble global mass matrix
SparseMatrix asm_global_mass_matr(const Mesh& mesh, const FEType& fe_type) {
  // build a finite element object with the given type (P1, P2, ...)
  const std::unique_ptr<FE> fe = FE::build_fe_type(fe_type);
  const DOFHandler dofh{mesh, *fe};
  const int global_ndofs = dofh.get_global_ndofs();
  const int local_ndofs = fe->get_ndofs();

  const Eigen::Index n = static_cast<Eigen::Index>(global_ndofs);

  std::vector<Eigen::Triplet<double>> triplets;

  //   Each triangle contributes at most ndofs*ndofs entries (9 for linear P1)

  triplets.reserve(mesh.elements.size() * local_ndofs * local_ndofs);

  for (std::size_t e = 0; e < mesh.elements.size(); e++) {
    const Element& element = mesh.elements.at(e);

    if (element.type == ElementType::Triangle3) {
      const auto el_nodes = get_element_nodes(element, mesh);
      const auto el_dofs = dofh.get_element_dof_indices(e);
      const FEMap2D tr_element(element, el_nodes, *fe);

      const MatrixXd local_mass_matrix = gen_local_mass_matr(tr_element, *fe);

      for (std::size_t i = 0; i < el_dofs.size(); i++) {
        for (std::size_t j = 0; j < el_dofs.size(); j++) {
          const std::size_t I = el_dofs.at(i);
          const std::size_t J = el_dofs.at(j);

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
SparseMatrix asm_global_stiffness_matr(const Mesh& mesh,
                                       const FEType& fe_type) {
  // build a finite element object with the given type (P1, P2, ...)
  const std::unique_ptr<FE> fe = FE::build_fe_type(fe_type);
  const DOFHandler dofh{mesh, *fe};
  const int global_ndofs = dofh.get_global_ndofs();
  const int local_ndofs = fe->get_ndofs();

  const Eigen::Index n = static_cast<Eigen::Index>(global_ndofs);

  std::vector<Eigen::Triplet<double>> triplets;

  //   Each triangle contributes at most ndofs*ndofs entries (9 for linear P1)

  triplets.reserve(mesh.elements.size() * local_ndofs * local_ndofs);

  for (std::size_t e = 0; e < mesh.elements.size(); e++) {
    const Element& element = mesh.elements.at(e);

    if (element.type == ElementType::Triangle3) {
      const auto el_nodes = get_element_nodes(element, mesh);
      const auto el_dofs = dofh.get_element_dof_indices(e);
      const FEMap2D tr_element(element, el_nodes, *fe);

      const MatrixXd local_stiffness_matrix =
          gen_local_stiffness_matr(tr_element, *fe);

      for (std::size_t i = 0; i < el_dofs.size(); i++) {
        for (std::size_t j = 0; j < el_dofs.size(); j++) {
          const std::size_t I = el_dofs.at(i);
          const std::size_t J = el_dofs.at(j);

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
VectorXd gen_local_vec(const FEMap2D& mapping,
                       STFunction func,
                       const FE& element,
                       double t) {
  const int n_dofs = element.get_ndofs();

  VectorXd lv = VectorXd::Zero(n_dofs);

  auto phys_points = mapping.get_phys_coords();
  const double detJ = std::abs(mapping.det_jacobian());
  std::vector<std::vector<double>> bfs = element.bfs_at_quad(quad_nodes);

  for (std::size_t q = 0; q < quad_nodes.size(); q++) {
    double f_val = func(phys_points.at(q).x, phys_points.at(q).y, t);

    for (int i = 0; i < n_dofs; i++) {
      lv(i) += detJ * quad_weights.at(q) * bfs.at(i).at(q) * f_val;
    }
  }

  return lv;
}

// assemble global load vector
VectorXd asm_global_vec(const Mesh& mesh, STFunction func, double t) {
  const P1_FE P1_element{};
  const DOFHandler dofh{mesh, P1_element};
  const int global_ndofs = dofh.get_global_ndofs();

  VectorXd gl_vector = VectorXd::Zero(global_ndofs);

  for (std::size_t e = 0; e < mesh.elements.size(); e++) {
    const Element& element = mesh.elements.at(e);

    if (element.type == ElementType::Triangle3) {
      const auto el_nodes = get_element_nodes(element, mesh);
      const auto el_dofs = dofh.get_element_dof_indices(e);
      const FEMap2D mapping(element, el_nodes, P1_element);

      VectorXd lv = gen_local_vec(mapping, func, P1_element, t);

      for (std::size_t i = 0; i < el_dofs.size(); ++i) {
        const std::size_t I = el_dofs.at(i);

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
    for (SparseMatrix::InnerIterator it(A, k); it; ++it) {
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

      //   b_row -= A_col,row * u_D(col)
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

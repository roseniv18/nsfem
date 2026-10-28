#include "assemble.h"

namespace nsfem::assembly {

// generate local stiffness matrix
MatrixXd gen_local_stiffness_matr(const FEMap2D& mapping, const FE& element) {
  const int ndofs = element.get_ndofs();

  MatrixXd ls_matrix = MatrixXd::Zero(ndofs, ndofs);

  for (std::size_t q = 0; q < quad_nodes.size(); q++) {
    const RefNode& pt = quad_nodes.at(q);

    const auto ref_grads = element.evaluate_grad_bfs(pt);

    const double detJ = std::abs(mapping.det_jacobian(pt));
    const MatrixXd JinvT = mapping.jacobianInvT(pt);

    // Physical gradients' values at this quadrature point (xi, eta)
    std::vector<PhysNode> phys_grads(ndofs);

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

  for (std::size_t q = 0; q < quad_nodes.size(); q++) {
    const RefNode& pt = quad_nodes.at(q);

    const double detJ = std::abs(mapping.det_jacobian(pt));

    const auto bfs = element.evaluate_bfs(pt);

    for (int i = 0; i < ndofs; i++) {
      for (int j = 0; j < ndofs; j++) {
        lm_matrix(i, j) += detJ * quad_weights.at(q) * bfs.at(i) * bfs.at(j);
      }
    }
  }

  return lm_matrix;
}

// assemble global mass matrix
SparseMatrix asm_mass_matrix(const Mesh& mesh,
                             const FE& fe,
                             const DOFHandler& dofh) {
  const std::size_t global_ndofs = dofh.get_global_ndofs();
  const int local_ndofs = fe.get_ndofs();

  const Eigen::Index n = static_cast<Eigen::Index>(global_ndofs);

  std::vector<Eigen::Triplet<double>> triplets;

  //   Each triangle contributes at most ndofs*ndofs entries (9 for linear P1)

  triplets.reserve(mesh.elements.size() * local_ndofs * local_ndofs);

  for (std::size_t e = 0; e < mesh.elements.size(); e++) {
    const Element& element = mesh.elements.at(e);

    if (!is_triangle_type(element.type))
      continue;

    const auto el_nodes = get_element_nodes(element, mesh);
    const auto el_dofs = dofh.get_element_dof_indices(e);
    const auto fe_geo = FE::build_geo_type(element.type);
    const FEMap2D tr_element(element, el_nodes, *fe_geo);

    const MatrixXd local_mass_matrix = gen_local_mass_matr(tr_element, fe);

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
SparseMatrix asm_stiffness_matrix(const Mesh& mesh,
                                  const FE& fe,
                                  const DOFHandler& dofh) {
  const std::size_t global_ndofs = dofh.get_global_ndofs();
  const int local_ndofs = fe.get_ndofs();

  const Eigen::Index n = static_cast<Eigen::Index>(global_ndofs);

  std::vector<Eigen::Triplet<double>> triplets;

  //   Each triangle contributes at most ndofs*ndofs entries (9 for linear P1)

  triplets.reserve(mesh.elements.size() * local_ndofs * local_ndofs);

  for (std::size_t e = 0; e < mesh.elements.size(); e++) {
    const Element& element = mesh.elements.at(e);

    if (!is_triangle_type(element.type))
      continue;

    const auto el_nodes = get_element_nodes(element, mesh);
    const auto el_dofs = dofh.get_element_dof_indices(e);
    const auto fe_geo = FE::build_geo_type(element.type);
    const FEMap2D tr_element(element, el_nodes, *fe_geo);

    const MatrixXd local_stiffness_matrix =
        gen_local_stiffness_matr(tr_element, fe);

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

  std::vector<std::vector<double>> bfs = element.bfs_at_quad(quad_nodes);

  for (std::size_t q = 0; q < quad_nodes.size(); q++) {
    const RefNode& pt = quad_nodes.at(q);

    auto phys_point = mapping.physical_point(pt);

    const double detJ = std::abs(mapping.det_jacobian(pt));
    double f_val = func(phys_point.x, phys_point.y, t);

    for (int i = 0; i < n_dofs; i++) {
      lv(i) += detJ * quad_weights.at(q) * bfs.at(i).at(q) * f_val;
    }
  }

  return lv;
}

// assemble global load vector
VectorXd asm_global_vec(const Mesh& mesh,
                        const FE& fe,
                        const DOFHandler& dofh,
                        STFunction func,
                        double t) {
  const std::size_t global_ndofs = dofh.get_global_ndofs();

  VectorXd gl_vector = VectorXd::Zero(global_ndofs);

  for (std::size_t e = 0; e < mesh.elements.size(); e++) {
    const Element& element = mesh.elements.at(e);

    if (!is_triangle_type(element.type))
      continue;

    const auto el_nodes = get_element_nodes(element, mesh);
    const auto el_dofs = dofh.get_element_dof_indices(e);
    const auto fe_geo = FE::build_geo_type(element.type);
    const FEMap2D mapping(element, el_nodes, *fe_geo);

    VectorXd lv = gen_local_vec(mapping, func, fe, t);

    for (std::size_t i = 0; i < el_dofs.size(); ++i) {
      const std::size_t I = el_dofs.at(i);

      gl_vector(I) += lv(i);
    }
  }

  return gl_vector;
}

MatrixXd gen_local_velocity_stiffness_matr(const FEMap2D& mapping,
                                           const FE& fe,
                                           const double viscosity = 1.0) {
  const int ndofs = fe.get_ndofs();

  // ndofs vector basis functions with 2 components each
  MatrixXd K = MatrixXd::Zero(2 * ndofs, 2 * ndofs);

  for (std::size_t q = 0; q < quad_nodes.size(); q++) {
    const RefNode& pt = quad_nodes.at(q);
    const double weight = quad_weights.at(q);

    const double detJ = std::abs(mapping.det_jacobian(pt));
    const auto grad_bfs = fe.evaluate_grad_bfs(pt);
    const auto JinvT = mapping.jacobianInvT(pt);

    // ref grad -> phys grad
    std::vector<Eigen::Vector2d> phys_grads(ndofs);
    for (int i = 0; i < ndofs; i++) {
      const Eigen::Vector2d ref_grad(grad_bfs.at(i).at(0),
                                     grad_bfs.at(i).at(1));

      phys_grads.at(i) = JinvT * ref_grad;
    }

    for (int i = 0; i < ndofs; i++) {
      for (int j = 0; j < ndofs; j++) {
        const double kij =
            viscosity * weight * detJ * phys_grads.at(i).dot(phys_grads.at(j));

        K(i, j) += kij;

        K(ndofs + i, ndofs + j) += kij;
      }
    }
  }

  return K;
}

MatrixXd gen_local_velocity_pressure_matr(const FEMap2D& mapping,
                                          const FE& velocity_fe,
                                          const FE& pressure_fe) {
  const int velocity_ndofs = velocity_fe.get_ndofs();
  const int pressure_ndofs = pressure_fe.get_ndofs();

  MatrixXd B = MatrixXd::Zero(2 * velocity_ndofs, pressure_ndofs);

  for (std::size_t q = 0; q < quad_nodes.size(); q++) {
    const RefNode pt = quad_nodes.at(q);
    const double weight = quad_weights.at(q);

    const auto vel_grad_bfs = velocity_fe.evaluate_grad_bfs(pt);
    const auto p_bfs = pressure_fe.evaluate_bfs(pt);

    const auto detJ = mapping.det_jacobian(pt);
    const auto JinvT = mapping.jacobianInvT(pt);

    std::vector<Eigen::Vector2d> phys_grads(velocity_ndofs);

    for (int i = 0; i < velocity_ndofs; i++) {
      const Eigen::Vector2d grad_ref(vel_grad_bfs.at(i).at(0),
                                     vel_grad_bfs.at(i).at(1));

      phys_grads.at(i) = JinvT * grad_ref;
    }

    for (int i = 0; i < velocity_ndofs; i++) {
      for (int a = 0; a < pressure_ndofs; a++) {
        B(i, a) += std::abs(detJ) * weight * p_bfs.at(a) * phys_grads.at(i).x();

        B(i + velocity_ndofs, a) +=
            std::abs(detJ) * weight * p_bfs.at(a) * phys_grads.at(i).y();
      }
    }
  }

  return B;
}

VectorXd gen_local_stokes_forcing_vector(const FEMap2D& mapping,
                                         const FE& velocity_fe,
                                         const VecFunction& forcing) {
  const int velocity_ndofs = velocity_fe.get_ndofs();

  VectorXd F = VectorXd::Zero(2 * velocity_ndofs);

  for (std::size_t q = 0; q < quad_nodes.size(); q++) {
    const RefNode& pt = quad_nodes.at(q);

    const double weight = quad_weights.at(q);

    const auto bfs = velocity_fe.evaluate_bfs(pt);

    const auto detJ = mapping.det_jacobian(pt);
    const PhysNode phys_point = mapping.physical_point(pt);

    const Eigen::Vector2d f = forcing(phys_point.x, phys_point.y);

    for (int i = 0; i < velocity_ndofs; i++) {
      F(i) += weight * std::abs(detJ) * bfs.at(i) * f.x();
      F(velocity_ndofs + i) += weight * std::abs(detJ) * bfs.at(i) * f.y();
    }
  }

  return F;
}

SparseMatrix asm_stokes_system(const Mesh& mesh,
                               const FE& velocity_fe,
                               const FE& pressure_fe,
                               const DOFHandler& velocity_dofh,
                               const DOFHandler& pressure_dofh,
                               VecFunction stokes_f_func) {
  const std::size_t velocity_ndofs = velocity_dofh.get_global_ndofs();
  const std::size_t pressure_ndofs = pressure_dofh.get_global_ndofs();
  const int local_velocity_ndofs = 2 * velocity_fe.get_ndofs();
  const int local_pressure_ndofs = pressure_fe.get_ndofs();

  const std::size_t global_ndofs = 2 * velocity_ndofs + pressure_ndofs;

  const Eigen::Index n = static_cast<Eigen::Index>(global_ndofs);

  std::vector<Eigen::Triplet<double>> triplets_vel_stiffness;
  std::vector<Eigen::Triplet<double>> triplets_vel_pressure_coupling;

  triplets_vel_stiffness.reserve(mesh.elements.size() * local_velocity_ndofs *
                                 local_velocity_ndofs);

  triplets_vel_pressure_coupling.reserve(
      mesh.elements.size() * local_velocity_ndofs * local_pressure_ndofs);

  //   Local velocity stiffness matrix
  MatrixXd lvs_matrix =
      MatrixXd::Zero(local_velocity_ndofs, local_velocity_ndofs);

  //   Local velocity-pressure coupling matrix
  MatrixXd lvp_matrix =
      MatrixXd::Zero(local_velocity_ndofs, local_pressure_ndofs);
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

}  // namespace nsfem::assembly

#include "heat_eq.h"

namespace nsfem::solvers {
using namespace nsfem::assembly;
using nsfem::convergence::h_initial_func;
using nsfem::fem::FE, nsfem::fem::DOFHandler;
using nsfem::mapping::FEMap2D;

HeatEq::HeatEq(const Mesh& mesh_,
               const FE& fe_,
               const DOFHandler& dofh_,
               double dt_,
               double T_,
               STFunction h_func_,
               STFunction h_dir_func_)
    : mesh(mesh_), fe(fe_), dofh(dofh_) {
  if (dt_ <= 1e-14 || T_ <= 1e-14)
    throw std::invalid_argument("Timestep dt and time T must be > 0");

  dt = dt_;
  T = T_;
  n_steps = static_cast<int>(T_ / dt_);
  h_func = std::move(h_func_);
  h_dir_func = std::move(h_dir_func_);

  const int global_ndofs = dofh.get_global_ndofs();

  initial_condition = interpolate_at_dofs(mesh, fe, dofh, h_initial_func, 0.0);

  rhs_vec.resize(global_ndofs);
}

void HeatEq::update_rhs(double t) {
  rhs_vec.setZero();

  for (std::size_t e = 0; e < mesh.elements.size(); e++) {
    const Element& element = mesh.elements.at(e);

    if (!is_triangle_type(element.type))
      continue;

    const auto element_nodes = get_element_nodes(element, mesh);
    const auto element_global_dofs = dofh.get_element_dof_indices(e);

    const auto geo_fe = FE::build_geo_type(element.type);

    FEMap2D mapping(element, element_nodes, *geo_fe);

    VectorXd local_rhs = VectorXd::Zero(element_global_dofs.size());

    const auto bfs_at_quad = fe.bfs_at_quad(quad_nodes);

    for (std::size_t q = 0; q < quad_nodes.size(); q++) {
      const RefNode& ref_point = quad_nodes.at(q);

      const auto phys_point = mapping.physical_point(ref_point);

      const double h = h_func(phys_point.x, phys_point.y, t);

      for (std::size_t i = 0; i < element_global_dofs.size(); i++) {
        local_rhs(i) += quad_weights.at(q) * mapping.det_jacobian(ref_point) *
                        h * bfs_at_quad.at(i).at(q);
      }
    }

    for (std::size_t i = 0; i < element_global_dofs.size(); i++) {
      rhs_vec(element_global_dofs.at(i)) += local_rhs(i);
    }
  }
}

VectorXd HeatEq::solve() {
  // Assemble matrices
  SparseMatrix K = asm_stiffness_matrix(mesh, fe, dofh);
  SparseMatrix M = asm_mass_matrix(mesh, fe, dofh);

  // Initial condition
  VectorXd u = initial_condition;

  const int global_ndofs = dofh.get_global_ndofs();

  std::cout << "global ndofs = " << global_ndofs << '\n';
  std::cout << "M       = " << M.rows() << " x " << M.cols() << '\n';
  std::cout << "K       = " << K.rows() << " x " << K.cols() << '\n';
  std::cout << "u       = " << u.size() << '\n';
  std::cout << "rhs_vec = " << rhs_vec.size() << '\n';
  std::cout << "nodes   = " << mesh.nodes.size() << '\n';

  //   Find Dirichlet nodes
  const auto is_dirichlet = get_dirichlet_dofs(mesh, dofh);

  // Build matrix for implicit Euler (once before loop)
  SparseMatrix A = M + K * dt;

  //   Keep original matrix for RHS Dirichlet boundary correction
  SparseMatrix A_bc = A;

  //   Apply BC to A_bc
  apply_dirichlet_bc_matr(A_bc, is_dirichlet);

  //   Initialize Eigen's CG solver
  Eigen::ConjugateGradient<SparseMatrix, Eigen::Lower | Eigen::Upper> cg;

  //   Time stepping (Implicit Euler)
  for (int i = 0; i < n_steps; i++) {
    const double t_next = (i + 1) * dt;

    // Compute Dirichlet values
    const auto dirichlet_values =
        get_dirichlet_values(mesh, fe, dofh, is_dirichlet, h_dir_func, t_next);

    update_rhs(t_next);

    // Compute rhs
    VectorXd rhs = M * u + dt * rhs_vec;

    // Apply BC to RHS, using original matrix
    apply_dirichlet_bc_vec(A, rhs, is_dirichlet, dirichlet_values);

    // Solve Au^{n+1} = rhs
    cg.compute(A_bc);

    if (cg.info() != Eigen::Success) {
      throw std::runtime_error("Eigen CG decomposition failed!");
    }

    VectorXd u_new = cg.solveWithGuess(rhs, u);

    if (cg.info() != Eigen::Success) {
      throw std::runtime_error("Eigen CG failed to converge!");
    }

    u = std::move(u_new);
  }

  return u;
}

}  // namespace nsfem::solvers

#include "heat_eq.h"

HeatEq::HeatEq(const Mesh& mesh,
               double dt_,
               double T_,
               STFunction h_func_,
               STFunction h_dir_func_) {
  if (dt_ <= 1e-14 || T_ <= 1e-14) {
    throw std::invalid_argument("Timestep dt and time T must be > 0");
  }

  dt = dt_;
  T = T_;
  n_steps = static_cast<int>(T_ / dt_);
  h_func = std::move(h_func_);
  h_dir_func = std::move(h_dir_func_);

  const std::size_t n = mesh.nodes.size();

  initial_condition.resize(n);
  rhs_vec.resize(n);

  for (std::size_t i = 0; i < n; i++) {
    const Node& node = mesh.nodes.at(i);
    initial_condition(i) = h_initial_func(node.x, node.y, 0.0);
  }
}

void HeatEq::update_rhs(const Mesh& mesh, double t) {
  for (std::size_t i = 0; i < mesh.nodes.size(); i++) {
    const Node& node = mesh.nodes.at(i);
    rhs_vec(i) = h_func(node.x, node.y, t);
  }
}

VectorXd HeatEq::solve(const Mesh& mesh) {
  // Assemble matrices
  SparseMatrix K = asm_global_stiffness_matr(mesh);
  SparseMatrix M = asm_global_mass_matr(mesh);

  // Initial condition
  VectorXd u = initial_condition;

  //   Find Dirichlet nodes
  const auto is_dirichlet = get_dirichlet_nodes(mesh);

  // Build matrix for implicit Euler (once before loop)
  SparseMatrix A = M + K * dt;

  //   Keep original matrix for RHS Dirichlet boundary correction
  SparseMatrix A_bc = A;

  //   Apply BC to A_bc
  apply_dirichlet_bc_matr(A_bc, is_dirichlet);

  //   Time stepping
  for (int i = 0; i < n_steps; i++) {
    const double t_new = (i + 1) * dt;
    const double t = i * dt;

    // Compute Dirichlet values
    const auto dirichlet_values =
        get_dirichlet_values(mesh, is_dirichlet, h_dir_func, t);

    update_rhs(mesh, t_new);

    // Compute rhs
    VectorXd rhs = M * u + dt * rhs_vec;

    // Apply BC to RHS, using original matrix
    apply_dirichlet_bc_vec(A, rhs, is_dirichlet, dirichlet_values);

    // Solve Au^{n+1} = rhs
    ConjugateGradient cg(A, rhs, u);
    VectorXd u_new = cg.solve();

    u = std::move(u_new);
  }

  return u;
}

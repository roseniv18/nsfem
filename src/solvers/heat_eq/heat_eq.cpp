#include "heat_eq.h"

HeatEq::HeatEq(const Mesh& mesh_,
               const FE& fe_,
               const DOFHandler& dofh_,
               double dt_,
               double T_,
               STFunction h_func_,
               STFunction h_dir_func_)
    : mesh(mesh_), fe(fe_), dofh(dofh_) {
  if (dt_ <= 1e-14 || T_ <= 1e-14) {
    throw std::invalid_argument("Timestep dt and time T must be > 0");
  }

  dt = dt_;
  T = T_;
  n_steps = static_cast<int>(T_ / dt_);
  h_func = std::move(h_func_);
  h_dir_func = std::move(h_dir_func_);

  const int global_ndofs = dofh.get_global_ndofs();
  const int vertex_ndofs = mesh.nodes.size();
  const int edge_dofs = mesh.unique_edges.size();

  initial_condition.resize(global_ndofs);
  rhs_vec.resize(global_ndofs);

  for (std::size_t i = 0; i < vertex_ndofs; i++) {
    const Node& node = mesh.nodes.at(i);
    initial_condition(i) = h_initial_func(node.x, node.y, 0.0);

    if (fe.type == FEType::P2) {
      for (int e = 0; e < edge_dofs; e++) {
        const Edge& edge = mesh.unique_edges.at(e);

        const Node& node1 = mesh.nodes.at(edge.node_id_1);
        const Node& node2 = mesh.nodes.at(edge.node_id_2);

        //   P2 DOF at the midpoint
        const double x = 0.5 * (node1.x + node2.x);
        const double y = 0.5 * (node1.y + node2.y);

        const std::size_t global_dof = vertex_ndofs + e;

        initial_condition(global_dof) = h_initial_func(x, y, 0.0);
      }
    }
  }
}

void HeatEq::update_rhs(double t) {
  const int global_ndofs = dofh.get_global_ndofs();

  const int vertex_ndofs = mesh.nodes.size();
  const int edge_dofs = mesh.unique_edges.size();

  for (int i = 0; i < vertex_ndofs; i++) {
    const Node& node = mesh.nodes.at(i);
    rhs_vec(i) = h_func(node.x, node.y, t);

    if (fe.type == FEType::P2) {
      for (int e = 0; e < edge_dofs; e++) {
        const Edge& edge = mesh.unique_edges.at(e);

        const Node& node1 = mesh.nodes.at(edge.node_id_1);
        const Node& node2 = mesh.nodes.at(edge.node_id_2);

        //   P2 DOF at the midpoint
        const double x = 0.5 * (node1.x + node2.x);
        const double y = 0.5 * (node1.y + node2.y);

        const std::size_t global_dof = vertex_ndofs + e;

        rhs_vec(global_dof) = h_func(x, y, t);
      }
    }
  }
}

VectorXd HeatEq::solve() {
  // Assemble matrices
  SparseMatrix K = asm_global_stiffness_matr(mesh, fe, dofh);
  SparseMatrix M = asm_global_mass_matr(mesh, fe, dofh);

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
  const auto is_dirichlet = get_dirichlet_nodes(mesh, global_ndofs, fe.type);

  // Build matrix for implicit Euler (once before loop)
  SparseMatrix A = M + K * dt;

  //   Keep original matrix for RHS Dirichlet boundary correction
  SparseMatrix A_bc = A;

  //   Apply BC to A_bc
  apply_dirichlet_bc_matr(A_bc, is_dirichlet);

  //   Initialize Eigen's CG solver
  Eigen::ConjugateGradient<SparseMatrix, Eigen::Lower | Eigen::Upper> cg;

  //   Time stepping
  for (int i = 0; i < n_steps; i++) {
    const double t_new = (i + 1) * dt;
    const double t = i * dt;

    // Compute Dirichlet values
    const auto dirichlet_values = get_dirichlet_values(
        mesh, is_dirichlet, h_dir_func, global_ndofs, fe.type, t);

    update_rhs(t);

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

#include "poisson.h"

Poisson::Poisson(const Mesh& mesh_,
                 const FE& fe_,
                 const DOFHandler& dofh_,
                 STFunction f_func_,
                 STFunction dir_func_)
    : mesh(mesh_), fe(fe_), dofh(dofh_) {
  f_func = std::move(f_func_);
  dir_func = std::move(dir_func_);

  rhs_vec.resize(dofh.get_global_ndofs());
}

VectorXd Poisson::solve() {
  const int global_ndofs = dofh.get_global_ndofs();

  // Assemble stiffness matrix
  SparseMatrix K = asm_global_stiffness_matr(mesh, fe, dofh);

  // Find Dirichlet nodes
  const auto is_dirichlet =
      get_dirichlet_nodes(mesh, dofh.get_global_ndofs(), fe.type);

  // Assemble RHS
  rhs_vec = asm_global_vec(mesh, fe, dofh, f_func, 0.0);

  //   Compute Dirichlet values
  const auto dirichlet_values = get_dirichlet_values(
      mesh, is_dirichlet, dir_func, dofh.get_global_ndofs(), fe.type, 0.0);

  // Apply Dirichlet BC to system
  apply_dirichlet_bc(K, rhs_vec, is_dirichlet, dirichlet_values);

  //   Solve with CG
  const VectorXd initial_guess = VectorXd::Zero(global_ndofs);
  Eigen::ConjugateGradient<SparseMatrix, Eigen::Lower | Eigen::Upper> cg;

  cg.compute(K);

  std::cout << "global ndofs = " << global_ndofs << '\n';
  std::cout << "initial_guess =       = " << initial_guess.size() << '\n';
  std::cout << "K       = " << K.rows() << " x " << K.cols() << '\n';
  std::cout << "rhs_vec = " << rhs_vec.size() << '\n';
  std::cout << "nodes   = " << mesh.nodes.size() << '\n';

  VectorXd solution = cg.solveWithGuess(rhs_vec, initial_guess);

  return solution;
}

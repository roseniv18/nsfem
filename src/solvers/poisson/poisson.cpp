#include "poisson.h"

Poisson::Poisson(const Mesh& mesh, STFunction f_func_, STFunction dir_func_) {
  const std::size_t n = mesh.nodes.size();

  f_func = std::move(f_func_);
  dir_func = std::move(dir_func_);

  rhs_vec.resize(n);
}

std::vector<double> Poisson::solve(const Mesh& mesh) {
  // Assemble stiffness matrix
  Matrix<double> K = asm_global_stiffness_matr(mesh);

  // Find Dirichlet nodes
  const auto is_dirichlet = get_dirichlet_nodes(mesh);

  // Assemble RHS
  rhs_vec = asm_global_vec(mesh, f_func, 0.0);

  //   Compute Dirichlet values
  const auto dirichlet_values =
      get_dirichlet_values(mesh, is_dirichlet, dir_func, 0.0);

  // Apply Dirichlet BC to system
  apply_dirichlet_bc(K, rhs_vec, is_dirichlet, dirichlet_values);

  //   Solve with CG
  const std::vector<double> initial_guess(mesh.nodes.size(), 0.0);
  ConjugateGradient cg(K, rhs_vec, initial_guess);

  return cg.solve();
}

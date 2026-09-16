#include "poisson.h"

Poisson::Poisson(const Mesh& mesh, STFunction f_func_, STFunction dir_func_) {
  const std::size_t n = mesh.nodes.size();

  f_func = std::move(f_func_);
  dir_func = std::move(dir_func_);

  rhs_vec.resize(n);
}

VectorXd Poisson::solve(const Mesh& mesh) {
  // Assemble stiffness matrix
  SparseMatrix K = asm_global_stiffness_matr(mesh);

  // Find Dirichlet nodes
  const auto is_dirichlet = get_dirichlet_nodes(mesh);

  // Assemble RHS
  rhs_vec = asm_global_vec(mesh, f_func, 0.0);

  //   Compute Dirichlet values
  const auto dirichlet_values =
      get_dirichlet_values(mesh, is_dirichlet, dir_func, 0.0);

  // Apply Dirichlet BC to system
  apply_dirichlet_bc(K, rhs_vec, is_dirichlet, dirichlet_values);

  for (Eigen::Index k = 0; k < K.outerSize(); ++k) {
    for (SparseMatrix::InnerIterator it(K, k); it; ++it) {
      if (!std::isfinite(it.value())) {
        std::cout << "Non-finite K entry at (" << it.row() << ", " << it.col()
                  << "): " << it.value() << '\n';
      }
    }
  }

  for (Eigen::Index i = 0; i < rhs_vec.size(); ++i) {
    if (!std::isfinite(rhs_vec(i))) {
      std::cout << "Non-finite RHS at " << i << ": " << rhs_vec(i) << '\n';
    }
  }

  double max_symmetry_error = 0.0;

  for (Eigen::Index i = 0; i < K.rows(); ++i) {
    for (Eigen::Index j = 0; j < K.cols(); ++j) {
      max_symmetry_error =
          std::max(max_symmetry_error, std::abs(K.coeff(i, j) - K.coeff(j, i)));
    }
  }

  //   Solve with CG
  const VectorXd initial_guess = VectorXd::Zero(mesh.nodes.size());
  Eigen::ConjugateGradient<SparseMatrix, Eigen::Lower | Eigen::Upper> cg;

  cg.compute(K);

  VectorXd solution = cg.solveWithGuess(rhs_vec, initial_guess);

  return solution;
}

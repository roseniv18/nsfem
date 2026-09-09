#include "implicit_euler.h"
#include "assemble/assemble.h"

std::vector<double> implicit_euler_step(
    Matrix<double>& M,
    Matrix<double>& K,
    const std::vector<double>& u,
    const std::vector<double> f,
    const std::unordered_map<int, double> dirichlet_values,
    double dt) {
  Matrix<double> A = M + K * dt;

  std::vector<double> rhs = M * u;

  for (std::size_t i = 0; i < rhs.size(); i++) {
    rhs[i] += dt * f[i];
  }

  apply_dirichlet_bc(A, rhs, dirichlet_values);

  ConjugateGradient cg(A, rhs);

  return cg.solve();
}

#include "conjugate_gradient.h"

inline int idx(int i, int j, int Nx) {
  return i * Nx + j;
}

ConjugateGradient::ConjugateGradient(const Matrix<double>& matrix_,
                                     const std::vector<double>& rhs_)
    : matrix(matrix_), rhs(rhs_), solution(std::vector<double>(rhs_.size())) {}

std::vector<double> ConjugateGradient::solve() {
  std::size_t size = rhs.size();
  std::vector<double> sol(size, 0.0);
  std::vector<double> sol_new(size, 0.0);
  std::vector<double> residual(size, 0.0), dir(size, 0.0), matdotdir(size, 0.0);

  //   Conjugate Gradient

  //  initial guess
  residual = rhs;
  dir = residual;

  double rr = 0.0;
  for (std::size_t i = 0; i < size; i++) {
    rr += residual[i] * residual[i];
  }

  const double rr0 = rr;

  for (std::size_t iter = 0; iter < max_iter; iter++) {
    if (rr <= tol2 * rr0) {
      std::cout << "Converged after " << iter << " iterations\n";
      break;
    }

    // A.p dot product (matrix * direction vector)
    for (std::size_t i = 0; i < size; i++) {
      matdotdir[i] = 0.0;

      for (std::size_t j = 0; j < size; j++) {
        matdotdir[i] += matrix(i, j) * dir[j];
      }
    }

    // p.A.p dot product (dir * matrix * dir)
    double pAp = 0.0;

    for (std::size_t i = 0; i < size; i++) {
      pAp += dir[i] * matdotdir[i];
    }

    // step
    const double alpha = rr / pAp;

    // update solution and residual
    for (std::size_t i = 0; i < size; i++) {
      sol_new[i] += alpha * dir[i];
      residual[i] -= alpha * matdotdir[i];
    }

    double rr_new = 0.0;
    for (std::size_t i = 0; i < size; i++) {
      rr_new += residual[i] * residual[i];
    }

    if (rr_new <= tol2 * rr0) {
      std::cout << "Converged after " << iter + 1 << " iterations\n";
      rr = rr_new;
      break;
    }

    // update direction vector
    const double beta = rr_new / rr;
    for (std::size_t i = 0; i < size; ++i) {
      dir[i] = residual[i] + beta * dir[i];
    }

    rr = rr_new;
  }

  for (int k = 0; k < size; k++)
    sol[k] = sol_new[k];

  if (rr > tol2 * rr0) {
    std::cout << "Failed to converge " << '\n';
  }

  solution = sol;
  return solution;
}

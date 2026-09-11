#include "conjugate_gradient.h"

ConjugateGradient::ConjugateGradient(const Matrix<double>& matrix_,
                                     const std::vector<double>& rhs_,
                                     const std::vector<double>& initial_guess)
    : matrix(matrix_),
      rhs(rhs_),
      solution(initial_guess),
      guess(initial_guess) {}

std::vector<double> ConjugateGradient::solve() {
  const std::size_t size = rhs.size();

  // Start from the provided initial guess
  std::vector<double> sol = guess;

  std::vector<double> residual(size, 0.0);
  std::vector<double> dir(size, 0.0);
  std::vector<double> matdotdir(size, 0.0);

  // r_0 = b - A*x_0
  for (std::size_t i = 0; i < size; ++i) {
    double Ax = 0.0;

    for (std::size_t j = 0; j < size; ++j) {
      Ax += matrix(i, j) * sol[j];
    }

    residual[i] = rhs[i] - Ax;
  }

  // p_0 = r_0
  dir = residual;

  double rr = 0.0;

  for (std::size_t i = 0; i < size; ++i) {
    rr += residual[i] * residual[i];
  }

  const double rr0 = rr;

  // Already converged?
  if (rr0 <= tol2) {
    std::cout << "Initial guess already converged\n";
    solution = sol;
    return solution;
  }

  for (std::size_t iter = 0; iter < max_iter; ++iter) {
    // A*p
    for (std::size_t i = 0; i < size; ++i) {
      matdotdir[i] = 0.0;

      for (std::size_t j = 0; j < size; ++j) {
        matdotdir[i] += matrix(i, j) * dir[j];
      }
    }

    // p^T A p
    double pAp = 0.0;

    for (std::size_t i = 0; i < size; ++i) {
      pAp += dir[i] * matdotdir[i];
    }

    const double alpha = rr / pAp;

    // x_{k+1} = x_k + alpha p_k
    // r_{k+1} = r_k - alpha A p_k
    for (std::size_t i = 0; i < size; ++i) {
      sol[i] += alpha * dir[i];
      residual[i] -= alpha * matdotdir[i];
    }

    double rr_new = 0.0;

    for (std::size_t i = 0; i < size; ++i) {
      rr_new += residual[i] * residual[i];
    }

    if (rr_new <= tol2 * rr0) {
      std::cout << "Converged after " << iter + 1 << " iterations\n";
      solution = sol;
      return solution;
    }

    const double beta = rr_new / rr;

    for (std::size_t i = 0; i < size; ++i) {
      dir[i] = residual[i] + beta * dir[i];
    }

    rr = rr_new;
  }

  std::cout << "Failed to converge\n";

  solution = sol;
  return solution;
}

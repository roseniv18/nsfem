#ifndef CONJUGATE_GRADIENT
#define CONJUGATE_GRADIENT

#include "linalg/matrix.h"

class ConjugateGradient {
 public:
  ConjugateGradient(const Matrix<double>& matrix,
                    const std::vector<double>& rhs,
                    const std::vector<double>& guess);
  std::vector<double> solve();

  int max_iter = 100;
  double tol2 = 1e-6 * 1e-6;

 private:
  Matrix<double> matrix;
  std::vector<double> rhs;
  std::vector<double> solution;
  std::vector<double> guess;
};

#endif

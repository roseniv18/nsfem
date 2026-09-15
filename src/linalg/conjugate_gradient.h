#ifndef CONJUGATE_GRADIENT
#define CONJUGATE_GRADIENT

#include <Eigen/Dense>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class ConjugateGradient {
 public:
  ConjugateGradient(const MatrixXd& matrix,
                    const VectorXd& rhs,
                    const VectorXd& guess);
  VectorXd solve();

  int max_iter = 100;
  double tol2 = 1e-6 * 1e-6;

 private:
  MatrixXd matrix;
  VectorXd rhs;
  VectorXd solution;
  VectorXd guess;
};

#endif

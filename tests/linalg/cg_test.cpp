#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>

#include "linalg/conjugate_gradient.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using SparseMatrix = Eigen::SparseMatrix<double>;

/**
 * Test custom CG on a small SPD system.
 *
 * A = [[4, 1],
 *      [1, 3]]
 *
 * b = [6, 7]
 *
 * x_exact = [1, 2]
 */
TEST(ConjugateGradientTest, SolveSmallSPDSystem) {
  MatrixXd A = MatrixXd::Zero(2, 2);

  A(0, 0) = 4.0;
  A(0, 1) = 1.0;
  A(1, 0) = 1.0;
  A(1, 1) = 3.0;

  VectorXd b(2);
  b << 6.0, 7.0;

  VectorXd initial_guess(2);
  initial_guess << 0.0, 0.0;

  ConjugateGradient cg(A, b, initial_guess);

  const VectorXd x = cg.solve();

  ASSERT_EQ(x.size(), 2);

  EXPECT_NEAR(x(0), 1.0, 1e-12);
  EXPECT_NEAR(x(1), 2.0, 1e-12);
}

/**
 * Test Eigen's sparse CG on the same SPD system.
 */
TEST(ConjugateGradientTest, EigenSolveSmallSPDSystem) {
  SparseMatrix A(2, 2);

  A.insert(0, 0) = 4.0;
  A.insert(0, 1) = 1.0;
  A.insert(1, 0) = 1.0;
  A.insert(1, 1) = 3.0;

  A.makeCompressed();

  VectorXd b(2);
  b << 6.0, 7.0;

  Eigen::ConjugateGradient<SparseMatrix, Eigen::Lower | Eigen::Upper> cg;

  cg.setTolerance(1e-12);
  cg.compute(A);

  ASSERT_EQ(cg.info(), Eigen::Success);

  const VectorXd x = cg.solve(b);

  ASSERT_EQ(cg.info(), Eigen::Success);
  ASSERT_EQ(x.size(), 2);

  EXPECT_NEAR(x(0), 1.0, 1e-12);
  EXPECT_NEAR(x(1), 2.0, 1e-12);
}

/**
 * Compare our custom CG against Eigen's sparse CG.
 *
 * Both solvers solve exactly the same mathematical system,
 * although the custom solver currently operates on MatrixXd
 * while Eigen operates on SparseMatrix.
 */
TEST(ConjugateGradientTest, CustomAndEigenAgree) {
  MatrixXd A_dense = MatrixXd::Zero(2, 2);

  A_dense(0, 0) = 4.0;
  A_dense(0, 1) = 1.0;
  A_dense(1, 0) = 1.0;
  A_dense(1, 1) = 3.0;

  SparseMatrix A_sparse(2, 2);

  A_sparse.insert(0, 0) = 4.0;
  A_sparse.insert(0, 1) = 1.0;
  A_sparse.insert(1, 0) = 1.0;
  A_sparse.insert(1, 1) = 3.0;

  A_sparse.makeCompressed();

  VectorXd b(2);
  b << 6.0, 7.0;

  VectorXd initial_guess = VectorXd::Zero(2);

  // Our CG.
  ConjugateGradient custom_cg(A_dense, b, initial_guess);

  const VectorXd x_custom = custom_cg.solve();

  // Eigen CG.
  Eigen::ConjugateGradient<SparseMatrix, Eigen::Lower | Eigen::Upper> eigen_cg;

  eigen_cg.setTolerance(1e-12);
  eigen_cg.compute(A_sparse);

  ASSERT_EQ(eigen_cg.info(), Eigen::Success);

  const VectorXd x_eigen = eigen_cg.solve(b);

  ASSERT_EQ(eigen_cg.info(), Eigen::Success);

  ASSERT_EQ(x_custom.size(), x_eigen.size());

  for (Eigen::Index i = 0; i < x_custom.size(); ++i) {
    EXPECT_NEAR(x_custom(i), x_eigen(i), 1e-12);
  }
}

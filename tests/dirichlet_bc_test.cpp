#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "assemble/assemble.h"

using Eigen::VectorXd;
using SparseMatrix = Eigen::SparseMatrix<double>;

/**
 * Test applying a single Dirichlet value to a system.
 *
 * Original matrix:
 *
 *     [ 2 -1  0 ]
 * A = [-1  2 -1 ]
 *     [ 0 -1  2 ]
 *
 * Original RHS:
 *
 *     [ 1 ]
 * b = [ 2 ]
 *     [ 3 ]
 *
 * We prescribe u0 = 5.
 *
 * Expected modified matrix:
 *
 *     [ 1  0  0 ]
 * A = [ 0  2 -1 ]
 *     [ 0 -1  2 ]
 *
 * Expected modified RHS:
 *
 *     [ 5 ]
 * b = [ 7 ]
 *     [ 3 ]
 */
TEST(DirichletBCTest, ApplySingleDirichletValue) {
  SparseMatrix K(3, 3);

  K.insert(0, 0) = 2.0;
  K.insert(0, 1) = -1.0;

  K.insert(1, 0) = -1.0;
  K.insert(1, 1) = 2.0;
  K.insert(1, 2) = -1.0;

  K.insert(2, 1) = -1.0;
  K.insert(2, 2) = 2.0;

  K.makeCompressed();

  VectorXd f(3);
  f << 1.0, 2.0, 3.0;

  std::vector<bool> is_dirichlet(3, false);
  is_dirichlet[0] = true;

  std::vector<double> dirichlet_vals(3, 0.0);
  dirichlet_vals[0] = 5.0;

  apply_dirichlet_bc(K, f, is_dirichlet, dirichlet_vals);

  // Modified matrix
  EXPECT_NEAR(K.coeff(0, 0), 1.0, 1e-12);
  EXPECT_NEAR(K.coeff(0, 1), 0.0, 1e-12);
  EXPECT_NEAR(K.coeff(0, 2), 0.0, 1e-12);

  EXPECT_NEAR(K.coeff(1, 0), 0.0, 1e-12);
  EXPECT_NEAR(K.coeff(1, 1), 2.0, 1e-12);
  EXPECT_NEAR(K.coeff(1, 2), -1.0, 1e-12);

  EXPECT_NEAR(K.coeff(2, 0), 0.0, 1e-12);
  EXPECT_NEAR(K.coeff(2, 1), -1.0, 1e-12);
  EXPECT_NEAR(K.coeff(2, 2), 2.0, 1e-12);

  // Modified RHS
  EXPECT_NEAR(f(0), 5.0, 1e-12);
  EXPECT_NEAR(f(1), 7.0, 1e-12);
  EXPECT_NEAR(f(2), 3.0, 1e-12);
}

/**
 * Test applying multiple Dirichlet values to a system.
 *
 * We prescribe u0 = 5 and u2 = 10.
 *
 * Expected modified matrix:
 *
 *     [ 1  0  0 ]
 * A = [ 0  2  0 ]
 *     [ 0  0  1 ]
 *
 * Expected modified RHS:
 *
 *     [ 5 ]
 * b = [17 ]
 *     [10 ]
 */
TEST(DirichletBCTest, AppliesMultipleDirichletValues) {
  SparseMatrix K(3, 3);

  K.insert(0, 0) = 2.0;
  K.insert(0, 1) = -1.0;

  K.insert(1, 0) = -1.0;
  K.insert(1, 1) = 2.0;
  K.insert(1, 2) = -1.0;

  K.insert(2, 1) = -1.0;
  K.insert(2, 2) = 2.0;

  K.makeCompressed();

  VectorXd f(3);
  f << 1.0, 2.0, 3.0;

  std::vector<bool> is_dirichlet(3, false);
  is_dirichlet[0] = true;
  is_dirichlet[2] = true;

  std::vector<double> dirichlet_vals(3, 0.0);
  dirichlet_vals[0] = 5.0;
  dirichlet_vals[2] = 10.0;

  apply_dirichlet_bc(K, f, is_dirichlet, dirichlet_vals);

  EXPECT_NEAR(K.coeff(0, 0), 1.0, 1e-12);
  EXPECT_NEAR(K.coeff(0, 1), 0.0, 1e-12);
  EXPECT_NEAR(K.coeff(0, 2), 0.0, 1e-12);

  EXPECT_NEAR(K.coeff(1, 0), 0.0, 1e-12);
  EXPECT_NEAR(K.coeff(1, 1), 2.0, 1e-12);
  EXPECT_NEAR(K.coeff(1, 2), 0.0, 1e-12);

  EXPECT_NEAR(K.coeff(2, 0), 0.0, 1e-12);
  EXPECT_NEAR(K.coeff(2, 1), 0.0, 1e-12);
  EXPECT_NEAR(K.coeff(2, 2), 1.0, 1e-12);

  EXPECT_NEAR(f(0), 5.0, 1e-12);
  EXPECT_NEAR(f(1), 17.0, 1e-12);
  EXPECT_NEAR(f(2), 10.0, 1e-12);
}

/**
 * Test that matrix modification preserves symmetry.
 */
TEST(DirichletBCTest, PreservesSymmetry) {
  SparseMatrix K(3, 3);

  K.insert(0, 0) = 2.0;
  K.insert(0, 1) = -1.0;

  K.insert(1, 0) = -1.0;
  K.insert(1, 1) = 2.0;
  K.insert(1, 2) = -1.0;

  K.insert(2, 1) = -1.0;
  K.insert(2, 2) = 2.0;

  K.makeCompressed();

  std::vector<bool> is_dirichlet(3, false);
  is_dirichlet[1] = true;

  apply_dirichlet_bc_matr(K, is_dirichlet);

  for (Eigen::Index i = 0; i < K.rows(); ++i) {
    for (Eigen::Index j = 0; j < K.cols(); ++j) {
      EXPECT_NEAR(K.coeff(i, j), K.coeff(j, i), 1e-12);
    }
  }
}

/**
 * Test RHS modification separately.
 *
 * The important point is that the original matrix is only read and is not
 * modified by apply_dirichlet_bc_vec().
 */
TEST(DirichletBCTest, ApplyDirichletValueToRhsOnly) {
  SparseMatrix K(3, 3);

  K.insert(0, 0) = 2.0;
  K.insert(0, 1) = -1.0;

  K.insert(1, 0) = -1.0;
  K.insert(1, 1) = 2.0;
  K.insert(1, 2) = -1.0;

  K.insert(2, 1) = -1.0;
  K.insert(2, 2) = 2.0;

  K.makeCompressed();

  const SparseMatrix K_original = K;

  VectorXd f(3);
  f << 1.0, 2.0, 3.0;

  std::vector<bool> is_dirichlet(3, false);
  is_dirichlet[0] = true;

  std::vector<double> dirichlet_vals(3, 0.0);
  dirichlet_vals[0] = 5.0;

  apply_dirichlet_bc_vec(K, f, is_dirichlet, dirichlet_vals);

  EXPECT_NEAR(f(0), 5.0, 1e-12);
  EXPECT_NEAR(f(1), 7.0, 1e-12);
  EXPECT_NEAR(f(2), 3.0, 1e-12);

  // Matrix must remain unchanged
  for (Eigen::Index i = 0; i < K.rows(); ++i) {
    for (Eigen::Index j = 0; j < K.cols(); ++j) {
      EXPECT_NEAR(K.coeff(i, j), K_original.coeff(i, j), 1e-12);
    }
  }
}

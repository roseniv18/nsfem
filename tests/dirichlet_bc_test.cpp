#include <gtest/gtest.h>
#include "assemble/assemble.h"
#include "linalg/matrix.h"

/** Test applying single Dirichlet value to system
 * We prescribe the Dirichlet DOF u0 = 5
 * to the global matrix [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]
 * and right-hand side global load vector [1, 2, 3]
 *
 * Expected modified global matrix is [[1, 0, 0], [0, 2, -1], [0, -1, 2]]
 * and modified global load vector [5, 7, 3]
 */
TEST(DirichletBCTest, ApplySingleDirichletValue) {
  {
    Matrix<double> K(3, 3);

    K(0, 0) = 2.0;
    K(0, 1) = -1.0;
    K(0, 2) = 0.0;

    K(1, 0) = -1.0;
    K(1, 1) = 2.0;
    K(1, 2) = -1.0;

    K(2, 0) = 0.0;
    K(2, 1) = -1.0;
    K(2, 2) = 2.0;

    std::vector<double> f{1.0, 2.0, 3.0};

    std::vector<bool> is_dirichlet(3, false);
    is_dirichlet[0] = true;

    std::vector<double> dirichlet_vals(3);
    dirichlet_vals[0] = 5.0;

    apply_dirichlet_bc(K, f, is_dirichlet, dirichlet_vals);

    // Modified matrix
    EXPECT_NEAR(K(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(K(0, 1), 0.0, 1e-12);
    EXPECT_NEAR(K(0, 2), 0.0, 1e-12);

    EXPECT_NEAR(K(1, 0), 0.0, 1e-12);
    EXPECT_NEAR(K(1, 1), 2.0, 1e-12);
    EXPECT_NEAR(K(1, 2), -1.0, 1e-12);

    EXPECT_NEAR(K(2, 0), 0.0, 1e-12);
    EXPECT_NEAR(K(2, 1), -1.0, 1e-12);
    EXPECT_NEAR(K(2, 2), 2.0, 1e-12);

    // Modified RHS
    EXPECT_NEAR(f[0], 5.0, 1e-12);
    EXPECT_NEAR(f[1], 7.0, 1e-12);
    EXPECT_NEAR(f[2], 3.0, 1e-12);
  }
}

/** Test applying mumltiple Dirichlet values to system
 * We prescribe the Dirichlet DOFs u0 = 5 and u2 = 10
 * to the global matrix [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]
 * and right-hand side global load vector [1, 2, 3]
 *
 * Expected modified global matrix is [[1, 0, 0], [0, 2, 0], [0, 0, 1]]
 * and modified global load vector [5, 17, 10]
 */
TEST(DirichletBCTest, AppliesMultipleDirichletValues) {
  Matrix<double> K(3, 3);

  K(0, 0) = 2.0;
  K(0, 1) = -1.0;
  K(0, 2) = 0.0;

  K(1, 0) = -1.0;
  K(1, 1) = 2.0;
  K(1, 2) = -1.0;

  K(2, 0) = 0.0;
  K(2, 1) = -1.0;
  K(2, 2) = 2.0;

  std::vector<double> f{1.0, 2.0, 3.0};

  std::vector<bool> is_dirichlet(3, false);
  is_dirichlet[0] = true;
  is_dirichlet[2] = true;

  std::vector<double> dirichlet_vals(3);
  dirichlet_vals[0] = 5.0;
  dirichlet_vals[2] = 10.0;

  apply_dirichlet_bc(K, f, is_dirichlet, dirichlet_vals);

  EXPECT_NEAR(K(0, 0), 1.0, 1e-12);
  EXPECT_NEAR(K(0, 1), 0.0, 1e-12);
  EXPECT_NEAR(K(0, 2), 0.0, 1e-12);

  EXPECT_NEAR(K(1, 0), 0.0, 1e-12);
  EXPECT_NEAR(K(1, 1), 2.0, 1e-12);
  EXPECT_NEAR(K(1, 2), 0.0, 1e-12);

  EXPECT_NEAR(K(2, 0), 0.0, 1e-12);
  EXPECT_NEAR(K(2, 1), 0.0, 1e-12);
  EXPECT_NEAR(K(2, 2), 1.0, 1e-12);

  EXPECT_NEAR(f[0], 5.0, 1e-12);
  EXPECT_NEAR(f[1], 17.0, 1e-12);
  EXPECT_NEAR(f[2], 10.0, 1e-12);
}

/** Test if the Dirichlet BC modification preserves symmetry of the global
 * matrix
 */
TEST(DirichletBCTest, PreservesSymmetry) {
  Matrix<double> K(3, 3);

  K(0, 0) = 2.0;
  K(0, 1) = -1.0;
  K(0, 2) = 0.0;

  K(1, 0) = -1.0;
  K(1, 1) = 2.0;
  K(1, 2) = -1.0;

  K(2, 0) = 0.0;
  K(2, 1) = -1.0;
  K(2, 2) = 2.0;

  std::vector<double> f{1.0, 2.0, 3.0};

  std::vector<bool> is_dirichlet(3);
  is_dirichlet[1] = true;

  apply_dirichlet_bc_matr(K, is_dirichlet);

  for (int i = 0; i < K.n; ++i) {
    for (int j = 0; j < K.m; ++j) {
      EXPECT_NEAR(K(i, j), K(j, i), 1e-12);
    }
  }
}

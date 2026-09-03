#include <gtest/gtest.h>
#include "linalg/conjugate_gradient.h"
#include "linalg/matrix.h"

/** Test CG on small SPD system
 * A = [[4, 1], [1, 3]]
 * b = [6, 7]
 * x_exact = [1, 2]
 */
TEST(ConjugateGradientTest, SolveSmallSPDSystem) {
  Matrix<double> A(2, 2);

  A(0, 0) = 4.0;
  A(0, 1) = 1.0;
  A(1, 0) = 1.0;
  A(1, 1) = 3.0;

  std::vector<double> b{6.0, 7.0};

  ConjugateGradient cg(A, b);

  const auto x = cg.solve();

  ASSERT_EQ(x.size(), 2);

  EXPECT_NEAR(x[0], 1.0, 1e-12);
  EXPECT_NEAR(x[1], 2.0, 1e-12);
}

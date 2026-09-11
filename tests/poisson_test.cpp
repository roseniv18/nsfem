#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <numbers>
#include <vector>

#include "assemble/assemble.h"
#include "convergence/convergence.h"
#include "helpers/helpers.h"
#include "linalg/conjugate_gradient.h"
#include "mesh/parser.h"

TEST(PoissonTest, L2Convergence) {
  const std::vector<int> resolutions{4, 8, 16, 32};

  std::vector<double> errors;

  for (const int n : resolutions) {
    Mesh mesh = make_unit_square_mesh(n);

    // Assemble system
    Matrix<double> K = asm_global_stiffness_matr(mesh);

    std::vector<double> rhs = asm_global_vec(mesh, func, 0.0);

    // Find Dirichlet nodes
    const auto is_dirichlet = get_dirichlet_nodes(mesh);

    // Compute Dirichlet values
    const auto dirichlet_values =
        get_dirichlet_values(mesh, is_dirichlet, dir_func, 0.0);

    // Apply BCs
    apply_dirichlet_bc(K, rhs, is_dirichlet, dirichlet_values);

    const std::vector<double> initial_guess(mesh.nodes.size(), 0.0);

    // Solve K u = f
    ConjugateGradient cg(K, rhs, initial_guess);

    const std::vector<double> solution = cg.solve();

    // Compute L2 error
    const double error = global_l2_err(mesh, solution, sol_func, 0.0);

    errors.push_back(error);

    std::cout << "n = " << n << ", L2 error = " << error << '\n';

    EXPECT_GT(error, 0.0);
  }

  // ------------------------------------------------------------
  // Check convergence rate
  // ------------------------------------------------------------

  for (std::size_t i = 1; i < errors.size(); ++i) {
    const double rate = std::log(errors[i - 1] / errors[i]) / std::log(2.0);

    EXPECT_GT(rate, 1.8);
    EXPECT_LT(rate, 2.2);
  }
}

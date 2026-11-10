#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <vector>

#include "FEM/assemble/assemble.h"
#include "FEM/convergence/convergence.h"
#include "FEM/finite_element/P1_element.h"
#include "FEM/finite_element/P2_element.h"
#include "FEM/finite_element/dof_handler.h"
#include "helpers/helpers.h"
#include "mesh/mesh.h"
#include "solvers/stokes/stokes.h"

TEST(StokesTest, P2P1) {
  Mesh mesh = make_unit_square_mesh(8);

  const nsfem::fem::P2_FE velocity_fe;
  const nsfem::fem::P1_FE pressure_fe;

  const nsfem::fem::DOFHandler velocity_dofh(mesh, velocity_fe);
  const nsfem::fem::DOFHandler pressure_dofh(mesh, pressure_fe);

  const double viscosity = 1.0;

  nsfem::solvers::Stokes stokes(mesh, velocity_fe, pressure_fe, velocity_dofh,
                                pressure_dofh, nsfem::convergence::stokes_func,
                                nsfem::convergence::stokes_dir_func, viscosity);

  const VectorXd solution = stokes.solve();

  const Eigen::Index nv = velocity_dofh.get_global_ndofs();
  const Eigen::Index np = pressure_dofh.get_global_ndofs();

  ASSERT_EQ(solution.size(), 2 * nv + np);
  ASSERT_TRUE(solution.allFinite());

  const VectorXd ux = solution.head(nv);
  const VectorXd uy = solution.segment(nv, nv);
  VectorXd p = solution.tail(np);

  const auto velocity_dirichlet =
      nsfem::mesh::get_dirichlet_dofs(mesh, velocity_dofh);

  const auto velocity_dirichlet_vals = nsfem::mesh::get_dirichlet_values(
      mesh, velocity_fe, velocity_dofh, velocity_dirichlet,
      nsfem::convergence::stokes_dir_func, 0.0);

  ASSERT_EQ(velocity_dirichlet.size(), nv);
  ASSERT_EQ(velocity_dirichlet_vals.size(), 2 * nv);

  for (Eigen::Index i = 0; i < nv; i++) {
    if (!velocity_dirichlet.at(i))
      continue;

    EXPECT_NEAR(solution(i), velocity_dirichlet_vals.at(i), test_dtol);
    EXPECT_NEAR(solution(nv + i), velocity_dirichlet_vals.at(i), test_dtol);
  }

  //   Align pressure gauges

  const Eigen::Index pinned_pressure_dof = 0.0;

  const auto first_triangle = std::find_if(
      mesh.elements.begin(), mesh.elements.end(), [](const auto& element) {
        return nsfem::mesh::is_triangle_type(element.type);
      });

  ASSERT_NE(first_triangle, mesh.elements.end());

  const double x_ref = mesh.nodes.at(first_triangle->node_ids.at(0)).x;

  p.array() += x_ref;

  EXPECT_NEAR(solution(2 * nv + pinned_pressure_dof), 0.0, test_dtol);

  //   Error and convergence
  const double Err_ux = nsfem::convergence::global_l2_err(
      mesh, velocity_fe, velocity_dofh, ux, nsfem::convergence::stokes_ux_func,
      0.0);

  const double Err_uy = nsfem::convergence::global_l2_err(
      mesh, velocity_fe, velocity_dofh, uy, nsfem::convergence::stokes_uy_func,
      0.0);

  const double Err_u = std::sqrt(Err_ux * Err_ux + Err_uy * Err_uy);

  const double Err_p =
      nsfem::convergence::global_l2_err(mesh, pressure_fe, pressure_dofh, p,
                                        nsfem::convergence::stokes_p_func, 0.0);

  EXPECT_TRUE(std::isfinite(Err_u));
  EXPECT_TRUE(std::isfinite(Err_p));

  EXPECT_GT(Err_u, 0.0);
  EXPECT_GT(Err_p, 0.0);
}

TEST(StokesTest, L2ConvergenceP2P1) {
  const std::vector<int> mesh_resolutions{4, 8, 16, 32};
  std::vector<double> velocity_errors;
  std::vector<double> pressure_errors;

  velocity_errors.reserve(4);
  pressure_errors.reserve(4);

  const nsfem::fem::P2_FE velocity_fe;
  const nsfem::fem::P1_FE pressure_fe;

  const double viscosity = 1.0;

  for (const int resolution : mesh_resolutions) {
    const Mesh& mesh = make_unit_square_mesh(resolution);

    const DOFHandler velocity_dofh(mesh, velocity_fe);
    const DOFHandler pressure_dofh(mesh, pressure_fe);

    const std::size_t nv = velocity_dofh.get_global_ndofs();
    const std::size_t np = pressure_dofh.get_global_ndofs();

    // solve stokes

    nsfem::solvers::Stokes stokes(
        mesh, velocity_fe, pressure_fe, velocity_dofh, pressure_dofh,
        nsfem::convergence::stokes_func, nsfem::convergence::stokes_dir_func,
        viscosity);

    const VectorXd solution = stokes.solve();

    // calculate error and append to error list

    const VectorXd ux = solution.head(nv);
    const VectorXd uy = solution.segment(nv, nv);
    VectorXd p = solution.tail(np);

    const auto velocity_dirichlet =
        nsfem::mesh::get_dirichlet_dofs(mesh, velocity_dofh);

    const auto velocity_dirichlet_vals = nsfem::mesh::get_dirichlet_values(
        mesh, velocity_fe, velocity_dofh, velocity_dirichlet,
        nsfem::convergence::stokes_dir_func, 0.0);

    ASSERT_EQ(velocity_dirichlet.size(), nv);
    ASSERT_EQ(velocity_dirichlet_vals.size(), 2 * nv);

    for (std::size_t i = 0; i < nv; i++) {
      if (!velocity_dirichlet.at(i))
        continue;

      const Eigen::Index I = static_cast<Eigen::Index>(i);
      const Eigen::Index nvI = static_cast<Eigen::Index>(nv + i);

      EXPECT_NEAR(solution(static_cast<Eigen::Index>(I)),
                  velocity_dirichlet_vals.at(i), test_dtol);
      EXPECT_NEAR(solution(nvI), velocity_dirichlet_vals.at(i), test_dtol);
    }

    //   Align pressure gauges

    const auto first_triangle = std::find_if(
        mesh.elements.begin(), mesh.elements.end(), [](const auto& element) {
          return nsfem::mesh::is_triangle_type(element.type);
        });

    ASSERT_NE(first_triangle, mesh.elements.end());

    const double x_ref = mesh.nodes.at(first_triangle->node_ids.at(0)).x;

    p.array() += x_ref;

    const double Err_ux = nsfem::convergence::global_l2_err(
        mesh, velocity_fe, velocity_dofh, ux,
        nsfem::convergence::stokes_ux_func, 0.0);

    const double Err_uy = nsfem::convergence::global_l2_err(
        mesh, velocity_fe, velocity_dofh, uy,
        nsfem::convergence::stokes_uy_func, 0.0);

    const double Err_u = std::sqrt(Err_ux * Err_ux + Err_uy * Err_uy);

    const double Err_p = nsfem::convergence::global_l2_err(
        mesh, pressure_fe, pressure_dofh, p, nsfem::convergence::stokes_p_func,
        0.0);

    velocity_errors.push_back(Err_u);
    pressure_errors.push_back(Err_p);
  }

  //   Convergence rate
  for (std::size_t i = 1; i < mesh_resolutions.size(); i++) {
    EXPECT_LT(pressure_errors.at(i), pressure_errors.at(i - 1));
    EXPECT_LT(velocity_errors.at(i), velocity_errors.at(i - 1));

    const double vel_rate =
        std::log(velocity_errors.at(i - 1) / velocity_errors.at(i)) /
        std::log(2.0);

    const double pres_rate =
        std::log(pressure_errors.at(i - 1) / pressure_errors.at(i)) /
        std::log(2.0);

    std::cerr << mesh_resolutions[i - 1] << " -> " << mesh_resolutions[i]
              << "\n  Velocity: " << velocity_errors.at(i - 1) << " -> "
              << velocity_errors.at(i) << ", rate = " << vel_rate
              << "\n  Pressure: " << pressure_errors.at(i - 1) << " -> "
              << pressure_errors.at(i) << ", rate = " << pres_rate << std::endl;

    // Expected asymptotic rates
    // P2 velocity L2: O(h^3)
    // P1 pressure L2: O(h^2)

    EXPECT_GT(vel_rate, 2.4);
    EXPECT_LT(vel_rate, 3.6);

    /**
     * For the pressure function f(x) = x
     * It seems there is superconvergence
     * The asymptotic rate is ~h^3.5
     */

    EXPECT_GT(pres_rate, 1.5);
    EXPECT_LT(pres_rate, 3.6);
  }
}

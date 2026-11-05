#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "FEM/assemble/assemble.h"
#include "FEM/finite_element/P1_element.h"
#include "FEM/finite_element/P2_element.h"
#include "FEM/finite_element/dof_handler.h"
#include "helpers/helpers.h"
#include "mesh/mesh.h"

using Eigen::SparseMatrix, Eigen::MatrixXd;
using nsfem::assembly::asm_stokes_matrix, nsfem::assembly::asm_stiffness_matrix,
    nsfem::assembly::asm_stokes_rhs, nsfem::assembly::asm_global_vec;
using nsfem::fem::DOFHandler, nsfem::fem::P1_FE, nsfem::fem::P2_FE;
using nsfem::mesh::Mesh;

/** The FE method applied on the Stokes problem
 * -nu*Delta(u) + grad(p) = f, 	Omega
 * div(u) = 0, 					Omega
 * u = 0, 						dOmega
 *
 * (u, f - 2D vectors, p - scalar)
 *
 * Results in the linear system:
 * AU - BP 		= 	F
 * 		B^TU 	= 	0
 *
 * In block form:
 * [[nu*A, -B], [B^T, 0]] . [U, P] = [F, 0]
 *
 * Where:
 * A - Global velocity stiffness matrix (2N_u x 2N_u)
 * B - Global velocity pressure coupling matrix (2N_u x N_p)
 * U - Velocity solution unknown coefficients
 * P - Pressure solution unknown coefficients
 * F - RHS
 *
 * The Global velocity stiffness matrix A is a block diagonal matrix:
 * A = [[K, 0], [0, K]]
 * Where K is the ordinary stiffness matrix (N_u x N_u)
 */

/** Note:
 * The Eigen API has a norm method for matrices:
 * matr.norm();
 * This gives the Frobenius norm
 * It is 0 IFF the matrix is the 0 matrix.
 */

TEST(GlobalStokesAssemblyTest, HasCorrectDimensions) {
  const Mesh& mesh = make_two_triangle_mesh();

  const P2_FE& velocity_fe{};
  const P1_FE& pressure_fe{};

  const DOFHandler& velocity_dofh{mesh, velocity_fe};
  const DOFHandler& pressure_dofh{mesh, pressure_fe};

  const double viscosity = 2.5;

  const std::size_t Nu = velocity_dofh.get_global_ndofs();
  const std::size_t Np = pressure_dofh.get_global_ndofs();

  const std::size_t N = 2 * Nu + Np;

  const SparseMatrix G = asm_stokes_matrix(
      mesh, velocity_fe, pressure_fe, velocity_dofh, pressure_dofh, viscosity);

  ASSERT_EQ(G.rows(), N);
  ASSERT_EQ(G.cols(), N);
}

TEST(GlobalStokesAssemblyTest, VelocityBlocksMatchScalarStiffness) {
  const Mesh& mesh = make_two_triangle_mesh();

  const P2_FE& velocity_fe{};
  const P1_FE& pressure_fe{};

  const DOFHandler& velocity_dofh{mesh, velocity_fe};
  const DOFHandler& pressure_dofh{mesh, pressure_fe};

  const double viscosity = 2.5;

  const std::size_t Nu = velocity_dofh.get_global_ndofs();

  const SparseMatrix G = asm_stokes_matrix(
      mesh, velocity_fe, pressure_fe, velocity_dofh, pressure_dofh, viscosity);

  const SparseMatrix K = asm_stiffness_matrix(mesh, velocity_fe, velocity_dofh);

  const MatrixXd Gxx = G.block(0, 0, Nu, Nu);
  const MatrixXd Gyy = G.block(Nu, Nu, Nu, Nu);
  const MatrixXd Gxy = G.block(0, Nu, Nu, Nu);
  const MatrixXd Gyx = G.block(Nu, 0, Nu, Nu);

  const MatrixXd K_dense(K);

  EXPECT_NEAR((Gxx - viscosity * K_dense).norm(), 0.0, test_dtol);
  EXPECT_NEAR((Gyy - viscosity * K_dense).norm(), 0.0, test_dtol);

  EXPECT_NEAR(Gxy.norm(), 0.0, test_dtol);
  EXPECT_NEAR(Gyx.norm(), 0.0, test_dtol);
}

TEST(GlobalStokesAssemblyTest, PressureCouplingMatrices) {
  const Mesh& mesh = make_two_triangle_mesh();

  const P2_FE& velocity_fe{};
  const P1_FE& pressure_fe{};

  const DOFHandler& velocity_dofh{mesh, velocity_fe};
  const DOFHandler& pressure_dofh{mesh, pressure_fe};

  const double viscosity = 2.5;

  const std::size_t Nu = velocity_dofh.get_global_ndofs();
  const std::size_t Np = pressure_dofh.get_global_ndofs();

  const SparseMatrix G = asm_stokes_matrix(
      mesh, velocity_fe, pressure_fe, velocity_dofh, pressure_dofh, viscosity);

  const Eigen::Index pressure_offset = static_cast<Eigen::Index>(2 * Nu);

  //   Extract block matrices

  //   Velocity-pressure coupling
  const MatrixXd negativeB = G.block(0, pressure_offset, pressure_offset, Np);
  const MatrixXd BTranspose = G.block(pressure_offset, 0, Np, pressure_offset);

  //   Pressure-pressure coupling (zero matrix)
  const MatrixXd ZeroBlock = G.block(pressure_offset, pressure_offset, Np, Np);

  // B = B^-T
  EXPECT_NEAR((negativeB - BTranspose.transpose()).norm(), 0.0, test_dtol);

  //   ZeroBlock = 0
  EXPECT_NEAR(ZeroBlock.norm(), 0.0, test_dtol);

  //   Coupling must not be omitted
  EXPECT_GT(negativeB.norm(), 0.0);

  //   Global Stokes matrix must be symmetric
  const auto GTranspose = static_cast<decltype(G)>(G.transpose());

  EXPECT_NEAR((G - GTranspose).norm(), 0.0, test_dtol);
}

TEST(GlobalStokesAssemblyTest, GlobalRHSMatchesScalarLoadVectors) {
  const Mesh& mesh = make_two_triangle_mesh();

  const P2_FE& velocity_fe{};
  const P1_FE& pressure_fe{};

  const DOFHandler& velocity_dofh{mesh, velocity_fe};
  const DOFHandler& pressure_dofh{mesh, pressure_fe};

  const std::size_t Nu = velocity_dofh.get_global_ndofs();
  const std::size_t Np = pressure_dofh.get_global_ndofs();

  const std::size_t N = 2 * Nu + Np;

  const VecFunction forcing = [](double x, double y) -> Eigen::Vector2d {
    return Eigen::Vector2d(1.0 + x, 2.0 - y);
  };

  const STFunction forcing_x = [](double x, double, double) { return 1.0 + x; };

  const STFunction forcing_y = [](double, double y, double) { return 2.0 - y; };

  const VectorXd rhs =
      asm_stokes_rhs(mesh, velocity_fe, velocity_dofh, pressure_dofh, forcing);

  //   Independently assemble scalar load vectors
  const VectorXd F_x =
      asm_global_vec(mesh, velocity_fe, velocity_dofh, forcing_x, 0.0);

  const VectorXd F_y =
      asm_global_vec(mesh, velocity_fe, velocity_dofh, forcing_y, 0.0);

  ASSERT_EQ(rhs.size(), N);

  //   Check velocity components
  EXPECT_NEAR((rhs.head(Nu) - F_x).norm(), 0.0, test_dtol);
  EXPECT_NEAR((rhs.segment(Nu, Nu) - F_y).norm(), 0.0, test_dtol);

  //   Pressure RHS part must be 0
  EXPECT_NEAR(rhs.tail(Np).norm(), 0.0, test_dtol);
}

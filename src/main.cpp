#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "FEM/convergence/convergence.h"
#include "FEM/finite_element/P1_element.h"
#include "FEM/finite_element/P2_element.h"
#include "mesh/parser.h"
#include "solvers/heat_eq/heat_eq.h"
#include "solvers/poisson/poisson.h"
#include "solvers/stokes/stokes.h"

using Eigen::VectorXd;

int main() {
  // --------------------------------------------------------------------------
  // Read mesh
  // --------------------------------------------------------------------------

  std::ifstream file("square.msh");

  if (!file.is_open()) {
    std::cerr << "Could not open square.msh\n";
    return 1;
  }

  std::string line;
  nsfem::parser::EntitySectionHeader entities_header{};
  nsfem::parser::EntityPhysicalTags entities{};
  std::map<PhysicalGroupKey, std::string> physical_groups;

  // $PhysicalNames and $Entities must be read before parsing elements.
  while (std::getline(file, line)) {
    if (line == "$PhysicalNames") {
      physical_groups = nsfem::parser::read_physical_names(file);
    } else if (line == "$Entities") {
      read_entity_block_header(file, entities_header);
      entities = read_entities(file, entities_header);
      break;
    }
  }

  Mesh mesh = read_mesh(file, entities, physical_groups);

  nsfem::parser::print_mesh(mesh);

  build_unique_edges_list(mesh);

  std::cout << "Number of elements: " << mesh.elements.size() << '\n';
  std::cout << "Number of unique edges: " << mesh.unique_edges.size() << '\n';
  std::cout << "Edge ids of element 213: "
            << mesh.elements.at(213).edge_ids.at(0) << " "
            << mesh.elements.at(213).edge_ids.at(1) << " "
            << mesh.elements.at(213).edge_ids.at(2) << '\n';

  file.close();

  // SET THE FINITE ELEMENT TYPE
  const std::unique_ptr<FE> fe = FE::build_fe_type(nsfem::fem::FEType::P2);
  // GENERATE DOFHANDLER
  const DOFHandler dofh(mesh, *fe);

  // --------------------------------------------------------------------------
  // Poisson
  // --------------------------------------------------------------------------

  std::cout << "========================================\n";
  std::cout << "Poisson equation\n";
  std::cout << "========================================\n";

  nsfem::solvers::Poisson poisson(mesh, *fe, dofh, nsfem::convergence::func,
                                  nsfem::convergence::dir_func);

  const VectorXd poisson_sol = poisson.solve();

  const VectorXd poisson_exact = nsfem::convergence::analytical_sol(
      mesh, *fe, dofh, nsfem::convergence::sol_func, 0.0);

  std::cout << "----------\n";
  std::cout << "L2 error = "
            << nsfem::convergence::global_l2_err(mesh, *fe, dofh, poisson_sol,
                                                 nsfem::convergence::sol_func,
                                                 0.0)
            << '\n';

  double poisson_max_error = 0.0;

  std::cout << "poisson_sol.size() = " << poisson_sol.size() << '\n';
  std::cout << "mesh.nodes.size() = " << mesh.nodes.size() << '\n';
  std::cout << "global ndofs = " << dofh.get_global_ndofs() << '\n';

  for (Eigen::Index i = 0; i < poisson_sol.size(); ++i) {
    poisson_max_error = std::max(poisson_max_error,
                                 std::abs(poisson_sol(i) - poisson_exact(i)));
  }

  std::cout << "Max nodal error = " << poisson_max_error << '\n';

  auto [poisson_min_it, poisson_max_it] =
      std::minmax_element(poisson_sol.begin(), poisson_sol.end());

  std::cout << "FEM min = " << *poisson_min_it << '\n';
  std::cout << "FEM max = " << *poisson_max_it << '\n';

  // --------------------------------------------------------------------------
  // Heat equation
  // --------------------------------------------------------------------------

  std::cout << "\n========================================\n";
  std::cout << "Heat equation\n";
  std::cout << "========================================\n";

  constexpr double dt = 0.01;
  constexpr double T = 1.0;

  nsfem::solvers::HeatEq heat_eq(mesh, *fe, dofh, dt, T,
                                 nsfem::convergence::h_func,
                                 nsfem::convergence::h_dir_func);

  const VectorXd heat_sol = heat_eq.solve();

  std::cout << "----------\n";
  std::cout << "L2 error = "
            << nsfem::convergence::global_l2_err(
                   mesh, *fe, dofh, heat_sol, nsfem::convergence::h_sol_func, T)
            << '\n';

  // --------------------------------------------------------------------------
  // Stokes equation (P2/P1 Taylor Hood)
  // --------------------------------------------------------------------------

  const nsfem::fem::P2_FE velocity_fe;
  const nsfem::fem::P1_FE pressure_fe;

  const DOFHandler velocity_dofh(mesh, velocity_fe);
  const DOFHandler pressure_dofh(mesh, pressure_fe);

  const double viscosity = 1.0;

  nsfem::solvers::Stokes stokes(mesh, velocity_fe, pressure_fe, velocity_dofh,
                                pressure_dofh, nsfem::convergence::stokes_func,
                                nsfem::convergence::stokes_dir_func, viscosity);

  const VectorXd stokes_sol = stokes.solve();

  const Eigen::Index nv = velocity_dofh.get_global_ndofs();
  const Eigen::Index np = pressure_dofh.get_global_ndofs();

  const VectorXd ux = stokes_sol.head(nv);
  const VectorXd uy = stokes_sol.segment(nv, nv);
  const VectorXd p = stokes_sol.tail(np);

  const auto first_triangle = std::find_if(
      mesh.elements.begin(), mesh.elements.end(), [](const auto& element) {
        return nsfem::mesh::is_triangle_type(element.type);
      });

  if (first_triangle == mesh.elements.end())
    throw std::runtime_error("No triangular elements found!");

  const double pressure_reference_x =
      mesh.nodes.at(first_triangle->node_ids.at(0)).x;

  VectorXd pressure_aligned = p;
  pressure_aligned.array() += pressure_reference_x;

  const double ux_l2_err = nsfem::convergence::global_l2_err(
      mesh, velocity_fe, velocity_dofh, ux, nsfem::convergence::stokes_ux_func,
      0.0);

  const double uy_l2_err = nsfem::convergence::global_l2_err(
      mesh, velocity_fe, velocity_dofh, uy, nsfem::convergence::stokes_uy_func,
      0.0);

  const double velocity_l2_err =
      std::sqrt(ux_l2_err * ux_l2_err + uy_l2_err * uy_l2_err);

  const double pressure_l2_err = nsfem::convergence::global_l2_err(
      mesh, pressure_fe, pressure_dofh, pressure_aligned,
      nsfem::convergence::stokes_p_func, 0.0);

  std::cout << "\n========================================\n";
  std::cout << "Stokes equation\n";
  std::cout << "========================================\n";

  std::cout << "Velocity DOFs (scalar): " << nv << '\n';
  std::cout << "Pressure DOFs: " << np << '\n';
  std::cout << "Total DOFs: " << stokes_sol.size() << '\n';

  std::cout << "----------\n";
  std::cout << "L2 error (ux): " << ux_l2_err << '\n';
  std::cout << "L2 error (uy): " << uy_l2_err << '\n';
  std::cout << "L2 error (velocity): " << velocity_l2_err << '\n';
  std::cout << "L2 error (pressure): " << pressure_l2_err << '\n';

  std::cout << "Pinned pressure DOF: " << p(0) << '\n';

  std::cout << "Expected L2 convergence: " << '\n';
  std::cout << "P2 Velocity: O(h^3) " << '\n';
  std::cout << "P1 Pressure: O(h^2)" << '\n';

  return 0;
}

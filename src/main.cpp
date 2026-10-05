#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "FEM/convergence/convergence.h"
#include "mesh/parser.h"
#include "solvers/heat_eq/heat_eq.h"
#include "solvers/poisson/poisson.h"

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
  EntitySectionHeader entities_header{};
  EntityPhysicalTags entities{};
  std::unordered_map<int, PhysicalGroup> physical_groups{};

  // $PhysicalNames and $Entities must be read before parsing elements.
  while (std::getline(file, line)) {
    if (line == "$PhysicalNames") {
      physical_groups = read_physical_names(file);
    } else if (line == "$Entities") {
      read_entity_block_header(file, entities_header);
      entities = read_entities(file, entities_header);
      break;
    }
  }

  Mesh mesh = read_mesh(file, entities, physical_groups);

  build_unique_edges_list(mesh);

  std::cout << "Number of elements: " << mesh.elements.size() << '\n';
  std::cout << "Number of unique edges: " << mesh.unique_edges.size() << '\n';
  std::cout << "Edge ids of element 213: "
            << mesh.elements.at(213).edge_ids.at(0) << " "
            << mesh.elements.at(213).edge_ids.at(1) << " "
            << mesh.elements.at(213).edge_ids.at(2) << '\n';

  file.close();

  // SET THE FINITE ELEMENT TYPE
  const std::unique_ptr<FE> fe = FE::build_fe_type(FEType::P2);
  // GENERATE DOFHANDLER
  const DOFHandler dofh(mesh, *fe);

  // --------------------------------------------------------------------------
  // Poisson
  // --------------------------------------------------------------------------

  std::cout << "========================================\n";
  std::cout << "Poisson equation\n";
  std::cout << "========================================\n";

  Poisson poisson(mesh, *fe, dofh, func, dir_func);

  const VectorXd poisson_sol = poisson.solve();

  const VectorXd poisson_exact = analytical_sol(mesh, *fe, dofh, sol_func, 0.0);

  std::cout << "----------\n";
  std::cout << "L2 error = "
            << global_l2_err(mesh, *fe, dofh, poisson_sol, sol_func, 0.0)
            << '\n';

  double poisson_max_error = 0.0;

  std::cout << "poisson_sol.size() = " << poisson_sol.size() << '\n';
  std::cout << "mesh.nodes.size() = " << mesh.nodes.size() << '\n';
  std::cout << "global ndofs = " << dofh.get_global_ndofs() << '\n';

  for (std::size_t i = 0; i < poisson_sol.size(); ++i) {
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

  constexpr double dt = 0.001;
  constexpr double T = 1.0;

  HeatEq heat_eq(mesh, *fe, dofh, dt, T, h_func, h_dir_func);

  const VectorXd heat_sol = heat_eq.solve();

  std::cout << "----------\n";
  std::cout << "L2 error = "
            << global_l2_err(mesh, *fe, dofh, heat_sol, h_sol_func, T) << '\n';

  //   double heat_max_error = 0.0;

  //   for (std::size_t i = 0; i < heat_sol.size(); ++i) {
  //     const Node& node = mesh.nodes.at(i);

  //     const double exact = h_sol_func(node.x, node.y, T);

  //     heat_max_error = std::max(heat_max_error, std::abs(heat_sol(i) -
  //     exact));
  //   }

  //   std::cout << "Max nodal error = " << heat_max_error << '\n';

  //   auto [heat_min_it, heat_max_it] =
  //       std::minmax_element(heat_sol.begin(), heat_sol.end());

  //   std::cout << "FEM min = " << *heat_min_it << '\n';
  //   std::cout << "FEM max = " << *heat_max_it << '\n';

  return 0;
}

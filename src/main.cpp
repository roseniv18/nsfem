#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "convergence/convergence.h"
#include "mesh/parser.h"
#include "solvers/heat_eq/heat_eq.h"
#include "solvers/poisson/poisson.h"

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

  file.close();

  // --------------------------------------------------------------------------
  // Poisson
  // --------------------------------------------------------------------------

  std::cout << "========================================\n";
  std::cout << "Poisson equation\n";
  std::cout << "========================================\n";

  Poisson poisson(mesh, func, dir_func);

  const std::vector<double> poisson_sol = poisson.solve(mesh);

  const std::vector<double> poisson_exact = analytical_sol(mesh);

  std::cout << "----------\n";
  std::cout << "L2 error = " << global_l2_err(mesh, poisson_sol, sol_func, 0.0)
            << '\n';

  double poisson_max_error = 0.0;

  for (std::size_t i = 0; i < poisson_sol.size(); ++i) {
    poisson_max_error = std::max(
        poisson_max_error, std::abs(poisson_sol.at(i) - poisson_exact.at(i)));
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
  constexpr double T = 0.1;

  HeatEq heat_eq(mesh, dt, T, h_func, h_dir_func);

  const std::vector<double> heat_sol = heat_eq.solve(mesh);

  std::cout << "----------\n";
  std::cout << "L2 error = " << global_l2_err(mesh, heat_sol, h_sol_func, T)
            << '\n';

  double heat_max_error = 0.0;

  for (std::size_t i = 0; i < heat_sol.size(); ++i) {
    const Node& node = mesh.nodes.at(i);

    const double exact = h_sol_func(node.x, node.y, T);

    heat_max_error = std::max(heat_max_error, std::abs(heat_sol.at(i) - exact));
  }

  std::cout << "Max nodal error = " << heat_max_error << '\n';

  auto [heat_min_it, heat_max_it] =
      std::minmax_element(heat_sol.begin(), heat_sol.end());

  std::cout << "FEM min = " << *heat_min_it << '\n';
  std::cout << "FEM max = " << *heat_max_it << '\n';

  return 0;
}

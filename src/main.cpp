#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "assemble/assemble.h"
#include "convergence/convergence.h"
#include "geometry/triangle_geometry.h"
#include "linalg/conjugate_gradient.h"
#include "mesh/parser.h"

using std::sin;
using std::numbers::pi;

int main() {
  std::ifstream file;

  file.open("square.msh");

  if (!file.is_open()) {
    std::cerr << "Could not open square.msh\n";
    return 1;
  }

  std::string line;
  EntitySectionHeader entities_header{};
  EntityPhysicalTags entities{};
  std::unordered_map<int, PhysicalGroup> physical_groups{};

  // entities must exists before elements are parsed
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
  //   print_mesh(mesh);

  file.close();

  std::cout << "----------" << '\n';
  std::cout << "Assembling global stiffness matrix..." << '\n';
  Matrix<double> gs_matrix = asm_global_stiffness_matr(mesh);

  std::cout << "----------" << '\n';
  std::cout << "Assembling global load vector..." << '\n';
  std::vector<double> gl_vector = asm_global_vec(mesh, func);

  auto dirichlet_nodes = get_dirichlet_nodes(mesh);
  auto dirichlet_values = get_dirichlet_values(mesh, dirichlet_nodes, dir_func);

  apply_dirichlet_bc(gs_matrix, gl_vector, dirichlet_values);

  std::cout << "----------" << '\n';
  std::cout << "Solving linear system..." << '\n';
  ConjugateGradient cg(gs_matrix, gl_vector);

  std::vector<double> fem_sol = cg.solve();
  std::vector<double> exact_sol = analytical_sol(mesh);

  double residual_norm_sq = 0.0;

  for (std::size_t i = 0; i < fem_sol.size(); ++i) {
    double Au_i = 0.0;

    for (std::size_t j = 0; j < fem_sol.size(); ++j) {
      Au_i += gs_matrix(i, j) * fem_sol[j];
    }

    const double r = gl_vector[i] - Au_i;

    residual_norm_sq += r * r;
  }

  std::cout << "Final residual norm = " << std::sqrt(residual_norm_sq) << '\n';

  auto [rhs_min_it, rhs_max_it] =
      std::minmax_element(gl_vector.begin(), gl_vector.end());

  std::cout << "RHS min = " << *rhs_min_it << '\n';
  std::cout << "RHS max = " << *rhs_max_it << '\n';

  auto [sol_min_it, sol_max_it] =
      std::minmax_element(fem_sol.begin(), fem_sol.end());

  std::cout << "Solution min = " << *sol_min_it << '\n';
  std::cout << "Solution max = " << *sol_max_it << '\n';

  std::cout << "----------" << '\n';
  std::cout << "L2 error is: " << global_l2_err(mesh, fem_sol) << '\n';

  double max_error = 0.0;

  for (std::size_t i = 0; i < fem_sol.size(); ++i) {
    max_error = std::max(max_error, std::abs(fem_sol.at(i) - exact_sol.at(i)));
  }
  std::cout << "----------" << '\n';
  std::cout << "Max nodal error = " << max_error << '\n';

  auto [min_it, max_it] = std::minmax_element(fem_sol.begin(), fem_sol.end());

  std::cout << "FEM min = " << *min_it << '\n';
  std::cout << "FEM max = " << *max_it << '\n';

  for (const Element& el : mesh.elements) {
    if (el.type == ElementType::Triangle3) {
      auto el_nodes = get_element_nodes(el, mesh);
      TriangleGEO triangle(el, el_nodes);

      local_vec lv = gen_local_vec(triangle, func);

      std::cout << "local RHS: " << lv[0] << " " << lv[1] << " " << lv[2]
                << '\n';

      const auto phys_points = triangle.get_phys_coords();

      for (std::size_t q = 0; q < quad_nodes.size(); ++q) {
        double f_val = func(phys_points[q]);

        std::cout << "q = " << q << " -> phys = (" << phys_points[q].x << ", "
                  << phys_points[q].y << ")"
                  << " f = " << f_val << " weight = " << quad_weights[q]
                  << " detJ = " << triangle.det_jacobian() << '\n';
      }

      for (const auto& node : el_nodes) {
        std::cout << "(" << node.x << ", " << node.y << ")\n";
      }

      break;
    }
  }

  return 0;
}

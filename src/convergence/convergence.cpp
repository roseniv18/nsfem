#include "convergence.h"

double element_l2_err_sq(const Element& element,
                         const Mesh& mesh,
                         const std::vector<double>& fem_sol,
                         STFunction exact_sol,
                         const double t) {
  std::vector<Node> tr_nodes = get_element_nodes(element, mesh);
  std::vector<double> u_vals(3);

  TriangleGEO tr(element, tr_nodes);

  u_vals.at(0) = fem_sol.at(element.node_indices.at(0));
  u_vals.at(1) = fem_sol.at(element.node_indices.at(1));
  u_vals.at(2) = fem_sol.at(element.node_indices.at(2));

  const auto quad_basis = bfs_at_quad();
  const auto phys_coords = tr.get_phys_coords();
  const auto detJ = tr.det_jacobian();

  double element_l2err_sq{};

  for (std::size_t q = 0; q < quad_nodes.size(); q++) {
    const double u_exact =
        exact_sol(phys_coords.at(q).x, phys_coords.at(q).y, t);

    const double u_h = u_vals.at(0) * quad_basis.at(0).at(q) +
                       u_vals.at(1) * quad_basis.at(1).at(q) +
                       u_vals.at(2) * quad_basis.at(2).at(q);

    const double err = u_exact - u_h;
    element_l2err_sq += detJ * quad_weights.at(q) * err * err;
  }

  return element_l2err_sq;
}

double global_l2_err(const Mesh& mesh,
                     const std::vector<double>& fem_sol,
                     STFunction exact_sol,
                     const double t) {
  double l2_error{};

  for (const Element& el : mesh.elements) {
    if (el.type == ElementType::Triangle3) {
      double element_l2_sq = element_l2_err_sq(el, mesh, fem_sol, exact_sol, t);
      l2_error += element_l2_sq;
    }
  }

  return std::sqrt(l2_error);
}

/** POISSON
 * Analytical solution of -𝚫u = 2π^2*sin(πx)*sin(πy)
 * u = 0, boundary
 * domain is unit square [0,1] X [0,1]
 */

// Dirichlet Function
double dir_func(const double x, const double y, const double t) {
  return 0.0;
}

// RHS Function
double func(const double x, const double y, const double t) {
  return 2 * pi * pi * sin(pi * x) * sin(pi * y);
}

// Analytical Solution Function
double sol_func(const double x, const double y, const double t) {
  return sin(pi * x) * sin(pi * y);
}

std::vector<double> analytical_sol(const Mesh& mesh) {
  std::vector<double> vec(mesh.nodes.size());

  for (std::size_t i = 0; i < mesh.nodes.size(); i++) {
    vec.at(i) = sol_func(mesh.nodes.at(i).x, mesh.nodes.at(i).y);
  }

  return vec;
}

/** HEAT EQUATION
 * Analytical solution of du/dt​ − Δu = 0
 * u(x,y,0) = sin(πx) * sin(πy) (initial condition)
 * u(.,.,t) = 0, dOmega (homogeneous Dirichlet BC)
 * domain is unit square [0,1] X [0,1]
 */

// Dirichlet Function
double h_dir_func(const double x, const double y, const double t) {
  return 0.0;
}

// RHS Function
double h_func(const double x, const double y, const double t) {
  return 0.0;
}

// Analytical Solution Function
double h_sol_func(const double x, const double y, const double t) {
  return sin(pi * x) * sin(pi * y) * exp(-2 * pi * pi * t);
}

// Initial Condition
double h_initial_func(const double x, const double y, const double t) {
  return sin(pi * x) * sin(pi * y);
}

#include "convergence.h"

namespace nsfem::convergence {
double element_l2_err_sq(const std::size_t element_id,
                         const Mesh& mesh,
                         const Element& element,
                         const FE& fe,
                         const DOFHandler& dofh,
                         const VectorXd& fem_sol,
                         STFunction exact_sol,
                         const double t) {
  if (!is_triangle_type(element.type)) {
    throw std::invalid_argument(
        "L2 error integration requires a triangular element.");
  }

  const std::vector<Node> element_nodes = get_element_nodes(element, mesh);

  const auto geometry_fe = FE::build_geo_type(element.type);
  const FEMap2D mapping(element, element_nodes, *geometry_fe);

  const std::size_t local_ndofs = fe.get_ndofs();

  const auto element_dofs = dofh.get_element_dof_indices(element_id);

  if (element_dofs.size() != local_ndofs) {
    throw std::runtime_error(
        "Local DOF count does not match the solution Finite Element.");
  }

  VectorXd u_local(static_cast<Eigen::Index>(local_ndofs));

  for (std::size_t i = 0; i < local_ndofs; i++) {
    u_local(static_cast<Eigen::Index>(i)) =
        fem_sol(static_cast<Eigen::Index>(element_dofs.at(i)));
  }

  const auto quad_basis = fe.bfs_at_quad(quad_nodes);

  double element_l2err_sq = 0.0;

  for (std::size_t q = 0; q < quad_nodes.size(); q++) {
    const RefNode& pt = quad_nodes.at(q);

    const auto phys_point = mapping.physical_point(pt);
    const double detJ = std::abs(mapping.det_jacobian(pt));

    const double u_exact = exact_sol(phys_point.x, phys_point.y, t);

    double u_h = 0.0;

    for (std::size_t i = 0; i < local_ndofs; i++) {
      u_h += u_local(static_cast<Eigen::Index>(i)) * quad_basis.at(i).at(q);
    }

    const double error = u_exact - u_h;

    element_l2err_sq += detJ * quad_weights.at(q) * error * error;
  }

  return element_l2err_sq;
}

double global_l2_err(const Mesh& mesh,
                     const FE& fe,
                     const DOFHandler& dofh,
                     const VectorXd& fem_sol,
                     STFunction exact_sol,
                     const double t) {
  double l2_error_sq = 0.0;

  for (std::size_t e = 0; e < mesh.elements.size(); e++) {
    const Element& el = mesh.elements.at(e);

    if (!is_triangle_type(el.type))
      continue;

    l2_error_sq +=
        element_l2_err_sq(e, mesh, el, fe, dofh, fem_sol, exact_sol, t);
  }

  return std::sqrt(l2_error_sq);
}

/** POISSON
 * Analytical solution of -𝚫u = 2π^2*sin(πx)*sin(πy)
 * u = 0, boundary
 * domain is unit square [0,1] X [0,1]
 */

// Dirichlet Function
// ? params: x,y,t
double dir_func(const double, const double, const double) {
  return 0.0;
}

// RHS Function
// ? params: x,y,t
double func(const double x, const double y, const double) {
  return 2 * pi * pi * sin(pi * x) * sin(pi * y);
}

// Analytical Solution Function
// ? params: x,y,t
double sol_func(const double x, const double y, const double) {
  return sin(pi * x) * sin(pi * y);
}

VectorXd analytical_sol(const Mesh& mesh,
                        const FE& fe,
                        const DOFHandler& dofh,
                        STFunction exact_sol,
                        const double t) {
  const std::size_t global_ndofs = dofh.get_global_ndofs();

  const std::size_t local_ndofs = fe.get_ndofs();

  const auto ref_nodes = fe.get_ref_nodes();

  if (ref_nodes.size() != local_ndofs) {
    throw std::runtime_error(
        "Reference node count does not match the solution Finite Element DOF "
        "count.");
  }

  VectorXd vec = VectorXd::Zero(static_cast<Eigen::Index>(global_ndofs));

  for (std::size_t e = 0; e < mesh.elements.size(); e++) {
    const Element& element = mesh.elements.at(e);

    if (!is_triangle_type(element.type))
      continue;

    const auto element_nodes = get_element_nodes(element, mesh);

    const auto geometry_fe = FE::build_geo_type(element.type);

    const FEMap2D mapping(element, element_nodes, *geometry_fe);

    const auto element_dofs = dofh.get_element_dof_indices(e);

    if (element_dofs.size() != local_ndofs) {
      throw std::runtime_error(
          "Local DOF count does not match the solution Finite Element DOF "
          "count.");
    }

    for (std::size_t i = 0; i < local_ndofs; i++) {
      const auto phys_point = mapping.physical_point(ref_nodes.at(i));

      const double value = exact_sol(phys_point.x, phys_point.y, t);

      const std::size_t global_dof = element_dofs.at(i);

      vec(static_cast<Eigen::Index>(global_dof)) = value;
    }
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
// ? params: x,y,t
double h_dir_func(const double, const double, const double) {
  return 0.0;
}

// RHS Function
// ? params: x,y,t
double h_func(const double, const double, const double) {
  return 0.0;
}

// Analytical Solution Function
// ? params: x,y,t
double h_sol_func(const double x, const double y, const double t) {
  return sin(pi * x) * sin(pi * y) * exp(-2 * pi * pi * t);
}

// Initial Condition
// ? params: x,y,t
double h_initial_func(const double x, const double y, const double) {
  return sin(pi * x) * sin(pi * y);
}

}  // namespace nsfem::convergence

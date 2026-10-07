#include "convergence.h"

double element_l2_err_sq(const std::size_t element_id,
                         const Mesh& mesh,
                         const Element& element,
                         const FE& fe,
                         const DOFHandler& dofh,
                         const VectorXd& fem_sol,
                         STFunction exact_sol,
                         const double t) {
  const std::vector<Node> tr_nodes = get_element_nodes(element, mesh);

  const FEMap2D mapping(element, tr_nodes, fe);

  const int global_ndofs = dofh.get_global_ndofs();
  const int local_ndofs = fe.get_ndofs();

  //   Get local -> global DOF indices
  const auto el_dofs = dofh.get_element_dof_indices(element_id);

  VectorXd u_local(local_ndofs);

  for (int i = 0; i < local_ndofs; i++) {
    u_local(i) = fem_sol(el_dofs.at(i));
  }

  const auto quad_basis = fe.bfs_at_quad(quad_nodes);
  const auto phys_coords = mapping.get_phys_coords();
  const auto detJ = mapping.det_jacobian();

  double element_l2err_sq = 0.0;

  for (std::size_t q = 0; q < quad_nodes.size(); q++) {
    const double u_exact =
        exact_sol(phys_coords.at(q).x, phys_coords.at(q).y, t);

    double u_h = 0.0;

    for (int i = 0; i < local_ndofs; i++) {
      u_h += u_local(i) * quad_basis.at(i).at(q);
    }

    const double err = u_exact - u_h;
    element_l2err_sq += std::abs(detJ) * quad_weights.at(q) * err * err;
  }

  return element_l2err_sq;
}

double global_l2_err(const Mesh& mesh,
                     const FE& fe,
                     const DOFHandler& dofh,
                     const VectorXd& fem_sol,
                     STFunction exact_sol,
                     const double t) {
  double l2_error = 0.0;

  for (std::size_t e = 0; e < mesh.elements.size(); e++) {
    const Element& el = mesh.elements.at(e);

    if (el.type == ElementType::Triangle3) {
      double element_l2_sq =
          element_l2_err_sq(e, mesh, el, fe, dofh, fem_sol, exact_sol, t);
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

VectorXd analytical_sol(const Mesh& mesh,
                        const FE& fe,
                        const DOFHandler& dofh,
                        STFunction exact_sol,
                        const double t) {
  const int global_ndofs = dofh.get_global_ndofs();
  VectorXd vec = VectorXd::Zero(global_ndofs);

  const std::size_t vertex_ndofs = mesh.nodes.size();
  const std::size_t edge_dofs = mesh.unique_edges.size();

  //   Vertex DOFs
  for (std::size_t i = 0; i < vertex_ndofs; i++) {
    const Node& node = mesh.nodes.at(i);

    vec(i) = exact_sol(node.x, node.y, t);
  }

  //   Edge DOFs
  if (fe.type == FEType::P2) {
    for (std::size_t e = 0; e < edge_dofs; e++) {
      const Edge& edge = mesh.unique_edges.at(e);

      const Node& node1 = mesh.nodes.at(edge.node_id_1);
      const Node& node2 = mesh.nodes.at(edge.node_id_2);

      //   P2 DOF at the midpoint
      const double x = 0.5 * (node1.x + node2.x);
      const double y = 0.5 * (node1.y + node2.y);

      const std::size_t global_dof = vertex_ndofs + e;

      vec(global_dof) = exact_sol(x, y, t);
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

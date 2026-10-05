#include "fe_mapping.h"

FEMap2D::FEMap2D(const Element& element,
                 const std::vector<Node>& nodes,
                 const FE& finite_element) {
  if (element.type != ElementType::Triangle3)
    throw std::invalid_argument("FEMap2D requires element of type Triangle3");

  const int n_dofs = finite_element.get_ndofs();
  const int n_quad = quad_nodes.size();

  phys_coords.resize(quad_nodes.size());
  phys_grads.resize(n_dofs);

  //   compute jacobian

  J(0, 0) = nodes.at(1).x - nodes.at(0).x;
  J(0, 1) = nodes.at(2).x - nodes.at(0).x;
  J(1, 0) = nodes.at(1).y - nodes.at(0).y;
  J(1, 1) = nodes.at(2).y - nodes.at(0).y;

  // compute determinant
  detJ = (J(0, 0) * J(1, 1)) - (J(0, 1) * J(1, 0));

  if (std::abs(detJ) <= 1e-14)
    throw std::runtime_error("Degenerate triangle (detJ ~= 0)");

  // compute JinvT
  JinvT(0, 0) = J(1, 1) / detJ;
  JinvT(0, 1) = -J(1, 0) / detJ;
  JinvT(1, 0) = -J(0, 1) / detJ;
  JinvT(1, 1) = J(0, 0) / detJ;

  // * compute physical coordinates and physical gradients

  for (std::size_t q = 0; q < n_quad; q++) {
    // basis functions at quadrature points
    // means: basis function i at quadrature node q
    const double xi = quad_nodes.at(q).at(0);
    const double eta = quad_nodes.at(q).at(1);

    const std::vector<double> N = finite_element.evaluate_bfs(xi, eta);
    const std::vector<std::vector<double>> gradN =
        finite_element.evaluate_grad_bfs(xi, eta);

    double N_geo[3] = {1.0 - xi - eta, xi, eta};
    for (std::size_t i = 0; i < 3; i++) {
      phys_coords.at(q).x += nodes.at(i).x * N_geo[i];
      phys_coords.at(q).y += nodes.at(i).y * N_geo[i];
    }

    for (std::size_t i = 0; i < n_dofs; i++) {
      phys_grads.at(i).x =
          JinvT(0, 0) * gradN.at(i).at(0) + JinvT(0, 1) * gradN.at(i).at(1);
      phys_grads.at(i).y =
          JinvT(1, 0) * gradN.at(i).at(0) + JinvT(1, 1) * gradN.at(i).at(1);
    }
  }
}

MatrixXd FEMap2D::jacobian() const {
  return J;
}

MatrixXd FEMap2D::jacobianInvT() const {
  return JinvT;
}

double FEMap2D::det_jacobian() const {
  return detJ;
}

std::vector<Point2D> FEMap2D::get_phys_grads() const {
  return phys_grads;
}

std::vector<Point2D> FEMap2D::get_phys_coords() const {
  return phys_coords;
}

#include "triangle_geometry.h"

TriangleGEO::TriangleGEO(const Element& element,
                         const std::vector<Node>& nodes) {
  if (element.type != ElementType::Triangle3)
    throw std::invalid_argument(
        "TriangleGEO requires element of type Triangle3");

  // compute jacobian
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

  // compute physical coordinates
  auto quad_basis = bfs_at_quad();

  for (std::size_t q = 0; q < 3; q++) {
    for (std::size_t i = 0; i < 3; i++) {
      // basis functions at quadrature points
      // means: basis function i at quadrature node q
      double N = quad_basis.at(i).at(q);

      phys_coords.at(q).x += nodes.at(i).x * N;
      phys_coords.at(q).y += nodes.at(i).y * N;
    }
  }

  // compute physical gradients
  auto ref_grads = basis_grads();

  for (int i = 0; i < 3; i++) {
    phys_grads.at(i).x =
        JinvT(0, 0) * ref_grads.at(i).x + JinvT(0, 1) * ref_grads.at(i).y;
    phys_grads.at(i).y =
        JinvT(1, 0) * ref_grads.at(i).x + JinvT(1, 1) * ref_grads.at(i).y;
  }
}

Matrix<double> TriangleGEO::jacobian() const {
  return J;
}

Matrix<double> TriangleGEO::jacobianInvT() const {
  return JinvT;
}

double TriangleGEO::det_jacobian() const {
  return detJ;
}

std::array<Point2D, 3> TriangleGEO::get_phys_grads() const {
  return phys_grads;
}

std::array<Point2D, 3> TriangleGEO::get_phys_coords() const {
  return phys_coords;
}

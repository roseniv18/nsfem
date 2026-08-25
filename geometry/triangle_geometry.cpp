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

  if (detJ <= 1e-14)
    throw std::runtime_error("Degenerate triangle (detJ ~= 0)");

  // compute JinvT
  JinvT(0, 0) = J(1, 1) / detJ;
  JinvT(0, 1) = -J(1, 0) / detJ;
  JinvT(1, 0) = -J(0, 1) / detJ;
  JinvT(1, 1) = J(0, 0) / detJ;

  // compute physical coordinates
  std::array<BasisFunction, 3> bfs = basis_functions();

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      const double N = bfs.at(i)(phys_coords[j]);

      phys_coords[j].x += nodes.at(i).x * N;
      phys_coords[j].y += nodes.at(i).y * N;
    }
  }

  // compute physical gradients
  std::array<Point2D, 3> ref_grads = basis_ref_grads();

  for (int i = 0; i < 3; i++) {
    phys_grads[i].x =
        JinvT(0, 0) * ref_grads[i].x + JinvT(0, 1) * ref_grads[i].y;
    phys_grads[i].y =
        JinvT(1, 0) * ref_grads[i].x + JinvT(1, 1) * ref_grads[i].y;
  }
}

Matrix<double> TriangleGEO::jacobian() const {
  return this->J;
}

Matrix<double> TriangleGEO::jacobianInvT() const {
  return this->JinvT;
}

double TriangleGEO::det_jacobian() const {
  return this->detJ;
}

std::array<Point2D, 3> TriangleGEO::get_phys_grads() const {
  return this->phys_grads;
}

std::array<Point2D, 3> TriangleGEO::get_phys_coords() const {
  return this->phys_coords;
}

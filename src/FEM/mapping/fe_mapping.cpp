#include "fe_mapping.h"

FEMap2D::FEMap2D(const Element& element_,
                 const std::vector<Node>& nodes_,
                 const FE& geometry_fe_)
    : element(element_), nodes(nodes_), geometry_fe(geometry_fe_) {}

PhysNode FEMap2D::physical_point(const RefNode& ref_pt) const {
  const auto N = geometry_fe.evaluate_bfs(ref_pt);
  PhysNode phys_pt{};

  for (std::size_t a = 0; a < N.size(); a++) {
    const Node& node = nodes.at(a);

    phys_pt.x += N.at(a) * node.x;
    phys_pt.y += N.at(a) * node.y;
  }

  return phys_pt;
}

Matrix2d FEMap2D::jacobian(const RefNode& pt) const {
  const auto gradN = geometry_fe.evaluate_grad_bfs(pt);
  Matrix2d J = Matrix2d::Zero();

  for (std::size_t i = 0; i < gradN.size(); i++) {
    const Node& node = nodes.at(i);

    J(0, 0) += gradN.at(i).at(0) * node.x;
    J(0, 1) += gradN.at(i).at(1) * node.x;

    J(1, 0) += gradN.at(i).at(0) * node.y;
    J(1, 1) += gradN.at(i).at(1) * node.y;
  }

  return J;
}

double FEMap2D::det_jacobian(const RefNode& pt) const {
  const auto detj = jacobian(pt).determinant();

  if (std::abs(detj) < 1e-14)
    throw std::runtime_error("Detected a degenerate triangle (detJ ~ 0)!");

  return detj;
}

Matrix2d FEMap2D::jacobianInvT(const RefNode& pt) const {
  const Matrix2d J = jacobian(pt);

  const double detJ = J.determinant();

  if (std::abs(detJ) < 1e-14)
    throw std::runtime_error("Degenerate element");

  return J.inverse().transpose();
}

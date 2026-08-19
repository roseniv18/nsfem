#include "affine.h"

AffineMap compute_affine(const Element& el, const std::vector<Node>& nodes) {
  // compute jacobian
  Matrix<double> J(2, 2);
  J(0, 0) = nodes[1].x - nodes[0].x;
  J(0, 1) = nodes[2].x - nodes[0].x;
  J(1, 0) = nodes[1].y - nodes[0].y;
  J(1, 1) = nodes[2].y - nodes[0].y;

  // compute determinant
  double detJ = (J(0, 0) * J(1, 1)) - (J(0, 1) * J(1, 0));

  // compute physical gradients
  Matrix<double> JinvT(2, 2);
  JinvT(0, 0) = J(1, 1) / detJ;
  JinvT(0, 1) = -J(1, 0) / detJ;
  JinvT(1, 0) = -J(0, 1) / detJ;
  JinvT(1, 1) = J(0, 0) / detJ;

  Point2D a_ph{}, b_ph{}, c_ph{};
  std::array<Point2D, 3> phys_grads = {a_ph, b_ph, c_ph};
  std::array<Point2D, 3> ref_grads = basis_ref_grads();

  for (int i = 0; i < 3; i++) {
    phys_grads[i].x =
        JinvT(0, 0) * ref_grads[i].x + JinvT(0, 1) * ref_grads[i].y;
    phys_grads[i].y =
        JinvT(1, 0) * ref_grads[i].x + JinvT(1, 1) * ref_grads[i].y;
  }

  AffineMap am{};
  am.Jacobian = J;
  am.detJ = detJ;
  am.phys_grads = phys_grads;

  return am;
}

Point2D map_to_phys(const Element& el,
                    const Point2D& pt,
                    const std::vector<Node>& nodes) {
  Point2D phys{};

  auto bfs = basis_functions();

  for (int i = 0; i < 3; i++) {
    const double N = bfs.at(i)(pt);

    phys.x += nodes[i].x * N;
    phys.y += nodes[i].y * N;
  }

  return phys;
}

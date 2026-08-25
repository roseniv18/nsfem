#include "affine.h"

AffineMap compute_affine(const std::vector<Node>& nodes) {
  std::cout << "compute_affine: nodes.size() = " << nodes.size() << '\n';
  // compute jacobian
  Matrix<double> J(2, 2);
  J(0, 0) = nodes.at(1).x - nodes.at(0).x;
  J(0, 1) = nodes.at(2).x - nodes.at(0).x;
  J(1, 0) = nodes.at(1).y - nodes.at(0).y;
  J(1, 1) = nodes.at(2).y - nodes.at(0).y;

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

Point2D map_to_phys(const Point2D& pt, const std::vector<Node>& nodes) {
  Point2D phys{};

  auto bfs = basis_functions();

  for (int i = 0; i < 3; i++) {
    const double N = bfs.at(i)(pt);

    phys.x += nodes.at(i).x * N;
    phys.y += nodes.at(i).y * N;
  }

  return phys;
}

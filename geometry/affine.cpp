#include "affine.h"

std::array<Gradient, 3> get_ref_grads() {
  Gradient a{-1, -1};
  Gradient b{1, 0};
  Gradient c{0, 1};

  return {a, b, c};
}

AffineMap compute_affine(const Element& element) {
  // compute jacobian
  std::array<std::array<double, 2>, 2> J;
  J[0][0] = element.nodes[1].x - element.nodes[0].x;
  J[0][1] = element.nodes[2].x - element.nodes[0].x;
  J[1][0] = element.nodes[1].y - element.nodes[0].y;
  J[1][1] = element.nodes[2].y - element.nodes[0].y;

  // compute determinant
  double detJ = (J[0][0] * J[1][1]) - (J[0][1] * J[1][0]);

  // compute physical gradients
  std::array<std::array<double, 2>, 2> JinvT;
  JinvT[0][0] = J[1][1] / detJ;
  JinvT[0][1] = -J[1][0] / detJ;
  JinvT[1][0] = -J[0][1] / detJ;
  JinvT[1][1] = J[0][0] / detJ;

  Gradient a_ph, b_ph, c_ph;
  std::array<Gradient, 3> phys_grads = {a_ph, b_ph, c_ph};
  std::array<Gradient, 3> ref_grads = get_ref_grads();

  for (int i = 0; i < 3; i++) {
    phys_grads[i].x =
        JinvT[0][0] * ref_grads[i].x + JinvT[0][1] * ref_grads[i].y;
    phys_grads[i].y =
        JinvT[1][0] * ref_grads[i].x + JinvT[1][1] * ref_grads[i].y;
  }

  AffineMap am;
  am.Jacobian = J;
  am.detJ = detJ;
  am.phys_grads = phys_grads;

  return am;
}

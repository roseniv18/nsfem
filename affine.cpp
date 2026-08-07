#include "affine.h"

std::array<Gradient, 3> get_ref_grads() {
  Gradient a{-1, -1};
  Gradient b{1, 0};
  Gradient c{0, 1};

  return {a, b, c};
}

AffineMap compute_affine(const Element& element, const Mesh& mesh) {
  // first get the element's node coordinates
  std::array<std::array<double, 2>, 3> nodes;

  for (int i = 0; i < element.node_indices.size(); i++) {
    nodes[i][0] = mesh.nodes.at(element.node_indices[i]).x;
    nodes[i][1] = mesh.nodes.at(element.node_indices[i]).y;
  }

  // compute jacobian
  std::array<std::array<double, 2>, 2> J;
  J[0][0] = nodes[1][0] - nodes[0][0];
  J[0][1] = nodes[2][0] - nodes[0][0];
  J[1][0] = nodes[1][1] - nodes[0][1];
  J[1][1] = nodes[2][1] - nodes[0][1];

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

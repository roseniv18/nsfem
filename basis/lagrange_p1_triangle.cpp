#include "lagrange_p1_triangle.h"

double phi1(const Point2D& p) {
  return 1 - p.x - p.y;
}

double phi2(const Point2D& p) {
  return p.x;
}

double phi3(const Point2D& p) {
  return p.y;
}

std::array<BasisFunction, 3> basis_functions() {
  return {phi1, phi2, phi3};
}

std::array<Point2D, 3> basis_ref_grads() {
  Point2D grad_phi1{-1, -1};
  Point2D grad_phi2{1, 0};
  Point2D grad_phi3{0, 1};

  return {grad_phi1, grad_phi2, grad_phi3};
}

std::array<std::array<double, 3>, 3> bfs_at_quad() {
  std::array<BasisFunction, 3> bfs = basis_functions();
  std::array<std::array<double, 3>, 3> quad_val{};

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      quad_val.at(i).at(j) = bfs.at(i)(quad_nodes.at(j));
    }
  }

  return quad_val;
}

std::array<std::array<Point2D, 3>, 3> brgrads_at_quad() {
  std::array<Point2D, 3> br_grads = basis_ref_grads();
  std::array<std::array<Point2D, 3>, 3> quad_val{};

  // constant for linear Lagrange triangles
  for (int i = 0; i < 3; i++) {
    quad_val.at(i) = br_grads;
  }

  return quad_val;
}

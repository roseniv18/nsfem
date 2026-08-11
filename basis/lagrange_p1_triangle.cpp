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

std::array<Point2D, 3> basis_gradients() {
  Point2D grad_phi1{-1, -1};
  Point2D grad_phi2{1, 0};
  Point2D grad_phi3{0, 1};

  return {grad_phi1, grad_phi2, grad_phi3};
}

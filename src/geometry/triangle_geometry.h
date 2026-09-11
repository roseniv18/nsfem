#ifndef TRIANGLE_GEOMETRY_H
#define TRIANGLE_GEOMETRY_H

#include "basis/lagrange_p1_triangle.h"
#include "linalg/matrix.h"
#include "mesh/mesh.h"
#include "quadrature/quadrature.h"

class TriangleGEO {
 public:
  TriangleGEO(const Element& element, const std::vector<Node>& nodes);

  //   access functions
  Matrix<double> jacobian() const;
  Matrix<double> jacobianInvT() const;
  double det_jacobian() const;
  std::array<Point2D, 3> get_phys_grads() const;
  std::array<Point2D, 3> get_phys_coords() const;

 private:
  Matrix<double> J{2, 2};
  Matrix<double> JinvT{2, 2};
  double detJ{};
  std::array<Point2D, 3> phys_grads{};
  std::array<Point2D, 3> phys_coords{};
};

#endif

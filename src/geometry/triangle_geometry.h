#ifndef TRIANGLE_GEOMETRY_H
#define TRIANGLE_GEOMETRY_H

#include <Eigen/Dense>
#include "basis/lagrange_p1_triangle.h"
#include "mesh/mesh.h"
#include "quadrature/quadrature.h"

using Eigen::MatrixXd, Eigen::VectorXd;

class TriangleGEO {
 public:
  TriangleGEO(const Element& element, const std::vector<Node>& nodes);

  //   access functions
  MatrixXd jacobian() const;
  MatrixXd jacobianInvT() const;
  double det_jacobian() const;
  std::array<Point2D, 3> get_phys_grads() const;
  std::array<Point2D, 3> get_phys_coords() const;

 private:
  MatrixXd J{2, 2};
  MatrixXd JinvT{2, 2};
  double detJ{};
  std::array<Point2D, 3> phys_grads{};
  std::array<Point2D, 3> phys_coords{};
};

#endif

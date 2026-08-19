#ifndef AFFINE_H
#define AFFINE_H

#include <array>
#include "basis/lagrange_p1_triangle.h"
#include "linalg/matrix.h"
#include "mesh/parser.h"

struct AffineMap {
  Matrix<double> Jacobian;
  double detJ;
  std::array<Point2D, 3> phys_grads;
};

// For now we naively assume element is of correct type (linear triangle)
// ? TODO: Implement Triangle struct with appropriate DOFs
// ? TODO: compute_affine to be compatible with different types of triangles
AffineMap compute_affine(const Element& element,
                         const std::vector<Node>& nodes);

Point2D map_to_phys(const Element& element,
                    const Point2D& point,
                    const std::vector<Node>& nodes);

#endif

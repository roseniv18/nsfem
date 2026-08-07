#ifndef AFFINE_H
#define AFFINE_H

#include <array>
#include "mesh/parser.h"

struct Gradient {
  double x;
  double y;
};

struct AffineMap {
  std::array<std::array<double, 2>, 2> Jacobian;
  double detJ;
  std::array<Gradient, 3> phys_grads;
};

std::array<Gradient, 3> get_ref_grads();

// For now we naively assume element is of correct type (linear triangle)
// ? TODO: Implement Triangle struct with appropriate DOFs
// ? TODO: compute_affine to be compatible with different types of triangles
AffineMap compute_affine(const Element& element, const Mesh& mesh);

#endif

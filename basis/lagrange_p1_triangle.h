#ifndef LAGRANGE_P1_TRIANGLE_H
#define LAGRANGE_P1_TRIANGLE_H

#include <array>
#include <functional>
#include "geometry/point2d.h"
#include "quadrature/quadrature.h"

using BasisFunction = std::function<double(const Point2D&)>;

double phi1(const Point2D& p);
double phi2(const Point2D& p);
double phi3(const Point2D& p);

std::array<BasisFunction, 3> basis_functions();
std::array<Point2D, 3> basis_ref_grads();

// evaluate each basis function at each quadrature point
std::array<std::array<double, 3>, 3> bfs_at_quad();

// evaluate each reference gradient at each quadrature point
std::array<std::array<Point2D, 3>, 3> brgrads_at_quad();

#endif

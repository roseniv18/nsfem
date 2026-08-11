#ifndef LAGRANGE_P1_TRIANGLE_H
#define LAGRANGE_P1_TRIANGLE_H

#include <array>
#include <functional>

struct Point2D {
  double x;
  double y;
};

using BasisFunction = std::function<double(const Point2D&)>;

double phi1(const Point2D& p);
double phi2(const Point2D& p);
double phi3(const Point2D& p);

std::array<BasisFunction, 3> basis_functions();
std::array<Point2D, 3> basis_gradients();

#endif

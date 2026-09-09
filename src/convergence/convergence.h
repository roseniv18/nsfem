#ifndef CONVERGENCE_H
#define CONVERGENCE_H

#include <vector>
#include "basis/lagrange_p1_triangle.h"
#include "geometry/triangle_geometry.h"
#include "mesh/parser.h"

using std::sin, std::numbers::pi;

double element_l2_err_sq(const Element& element,
                         const Mesh& mesh,
                         const std::vector<double>& fem_sol);

double global_l2_err(const Mesh& mesh, const std::vector<double>& fem_sol);

double dir_func(const Point2D& pt);
double func(const Point2D& pt);
double sol_func(const Point2D& pt);
std::vector<double> analytical_sol(const Mesh& mesh);

#endif

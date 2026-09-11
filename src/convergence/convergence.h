#ifndef CONVERGENCE_H
#define CONVERGENCE_H

#include <functional>
#include <vector>
#include "basis/lagrange_p1_triangle.h"
#include "geometry/triangle_geometry.h"
#include "math/function.h"
#include "mesh/mesh.h"

using std::sin, std::numbers::pi, std::exp;

double element_l2_err_sq(const Element& element,
                         const Mesh& mesh,
                         const std::vector<double>& fem_sol,
                         STFunction exact_sol,
                         const double t);

double global_l2_err(const Mesh& mesh,
                     const std::vector<double>& fem_sol,
                     STFunction exact_sol,
                     const double t);

//  POISSON
double dir_func(const double x, const double y, const double t = 0.0);
double func(const double x, const double y, const double t = 0.0);
double sol_func(const double x, const double y, const double t = 0.0);
std::vector<double> analytical_sol(const Mesh& mesh);

// HEAT
double h_dir_func(const double x, const double y, const double t);
double h_func(const double x, const double y, const double t);
double h_initial_func(const double x, const double y, const double t = 0.0);
double h_sol_func(const double x, const double y, const double t);

#endif

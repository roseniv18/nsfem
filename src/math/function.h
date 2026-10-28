#ifndef M_FUNCTION_H
#define M_FUNCTION_H

#include <Eigen/Dense>
#include <functional>

using STFunction =
    std::function<double(const double x, const double y, const double t)>;

using VecFunction = std::function<Eigen::Vector2d(double, double)>;

#endif

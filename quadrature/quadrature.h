#ifndef QUADRATURE_H
#define QUADRATURE_H

#include <array>
#include "geometry/point2d.h"

constexpr std::array<Point2D, 3> quad_nodes = {
    {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}}};

constexpr std::array<double, 3> quad_weights = {1.0 / 6.0, 1.0 / 6.0,
                                                1.0 / 6.0};

#endif

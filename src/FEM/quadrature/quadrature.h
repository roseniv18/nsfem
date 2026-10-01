#ifndef QUADRATURE_H
#define QUADRATURE_H

#include <vector>

const std::vector<std::vector<double>> quad_nodes = {
    {0.5, 0.0},
    {0.0, 0.5},
    {0.5, 0.5},
};

const std::vector<double> quad_weights = {1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0};

#endif

#ifndef QUADRATURE_H
#define QUADRATURE_H

#include <vector>

// const std::vector<std::vector<double>> quad_nodes = {
//     {0.5, 0.0},
//     {0.0, 0.5},
//     {0.5, 0.5},
// };

// const std::vector<double> quad_weights = {1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0};

const std::vector<std::vector<double>> quad_nodes = {
    // Center point
    {1.0 / 3.0, 1.0 / 3.0},

    // First symmetric group (Inner points)
    {0.059715871789770, 0.470142064105115},
    {0.470142064105115, 0.059715871789770},
    {0.470142064105115, 0.470142064105115},

    // Second symmetric group (Outer points near corners)
    {0.797426985353087, 0.101286507323456},
    {0.101286507323456, 0.797426985353087},
    {0.101286507323456, 0.101286507323456}};

const std::vector<double> quad_weights = {
    0.112500000000000,  // Center weight

    0.066197076394253,  // Inner group weights
    0.066197076394253, 0.066197076394253,

    0.062969590272414,  // Outer group weights
    0.062969590272414, 0.062969590272414};

#endif

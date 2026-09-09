#include <unordered_map>
#include <vector>
#include "linalg/conjugate_gradient.h"
#include "linalg/matrix.h"

/** Time-stepping for the 2D heat equation
 * The linear system to be solved at each time step is:
 * Au^(n+1) = b
 * where
 * A = M + dt*K
 * b = Mu^n + dt*f^(n+1)
 */
std::vector<double> implicit_euler_step(
    Matrix<double>& M,
    Matrix<double>& K,
    const std::vector<double>& u,
    const std::vector<double> f,
    const std::unordered_map<int, double> dirichlet_values,
    double dt);

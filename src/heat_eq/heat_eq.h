#include <cmath>
#include <vector>
#include "assemble/assemble.h"
#include "linalg/conjugate_gradient.h"
#include "mesh/parser.h"

using std::exp;
using std::sin;
using std::numbers::pi;

/** Solve the 2D heat equation on the unit square
 * du/dt​ − Δu = 0
 * u(x,y,0) = sin(πx) * sin(πy)
 * u(.,.,t) = 0, dOmega (homogeneous Dirichlet BC)
 */

class HeatEq {
 public:
  HeatEq(const Mesh& mesh, const double dt, double T);
  std::vector<double> solve(const Mesh& mesh);

 private:
  std::vector<double> initial_condition;
  std::vector<double> rhs_vec;
  std::vector<double> sol;
  double dt{};
  double T{};
  int n_steps{};
};

double u_initial(const Point2D& pt);
std::vector<double> initial_condition(const Mesh& mesh);

double exact_sol(const Point2D& pt, double t);
std::vector<double> exact_sol_vec(const Mesh& mesh, double t);

double f(const Point2D& pt);
std::vector<double> f_vec(const Mesh& mesh);

double dirichlet_func(const Point2D& pt);

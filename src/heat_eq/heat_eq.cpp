#include "heat_eq.h"

HeatEq::HeatEq(const Mesh& mesh, double dt_, double T_) {
  if (dt_ <= 1e-14 || T_ <= 1e-14) {
    throw std::invalid_argument("Timestep dt and time T must be > 0");
  }

  dt = dt_;
  T = T_;
  n_steps = static_cast<int>(T_ / dt_);

  std::size_t n = mesh.nodes.size();

  initial_condition.reserve(n);
  rhs_vec.reserve(n);
  sol.reserve(n);

  for (std::size_t i = 0; i < n; i++) {
    const Node& node = mesh.nodes.at(i);
    initial_condition.push_back(u_initial({node.x, node.y}));
    rhs_vec.push_back(f({node.x, node.y}));
  }
}

std::vector<double> HeatEq::solve(const Mesh& mesh) {
  // Assemble matrices
  Matrix<double> K = asm_global_stiffness_matr(mesh);
  Matrix<double> M = asm_global_mass_matr(mesh);

  // Initial condition
  std::vector<double> u = initial_condition;

  //   Find Dirichlet nodes
  const auto dirichlet_nodes = get_dirichlet_nodes(mesh);

  // Compute Dirichlet values
  const auto dirichlet_values =
      get_dirichlet_values(mesh, dirichlet_nodes, dirichlet_func);

  // Build matrix for implicit Euler (once before loop)
  Matrix<double> A = M + K * dt;

  //   Apply BC to A
  apply_dirichlet_bc_matr(A, dirichlet_values);

  //   Time stepping
  for (int i = 0; i < n_steps; i++) {
    // rhs = M * u^n
    std::vector<double> rhs = M * u;

    apply_dirichlet_bc_vec(A, rhs, dirichlet_values);

    // Solve Au^{n+1} = rhs
    ConjugateGradient cg(A, rhs);
    std::vector<double> u_new = cg.solve();

    u = std::move(u_new);
  }

  return u;
}

double u_initial(const Point2D& pt) {
  return sin(pi * pt.x) * sin(pi * pt.y);
}

std::vector<double> initial_condition(const Mesh& mesh) {
  std::vector<double> u(mesh.nodes.size());

  for (std::size_t i = 0; i < mesh.nodes.size(); i++) {
    const auto& node = mesh.nodes.at(i);

    u.at(i) = u_initial({node.x, node.y});
  }

  return u;
}

double exact_sol(const Point2D& pt, double t) {
  return exp(-2 * pi * pi * t) * sin(pi * pt.x) * sin(pi * pt.y);
}

std::vector<double> exact_sol_vec(const Mesh& mesh, double t) {
  std::vector<double> vec(mesh.nodes.size());

  for (std::size_t i = 0; i < mesh.nodes.size(); i++) {
    const auto& node = mesh.nodes.at(i);

    vec.at(i) = exact_sol({node.x, node.y}, t);
  }

  return vec;
}

double f(const Point2D& pt) {
  return 0.0;
}

std::vector<double> f_vec(const Mesh& mesh) {
  std::vector<double> vec(mesh.nodes.size());

  for (std::size_t i = 0; i < mesh.nodes.size(); i++) {
    const auto& node = mesh.nodes.at(i);

    vec.at(i) = f({node.x, node.y});
  }

  return vec;
}

double dirichlet_func(const Point2D& pt) {
  return 0.0;
}

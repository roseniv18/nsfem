#include "P2_element.h"

P2_FE::P2_FE()
    : dofs({{0, DOFLoc::Vertex, 0},
            {1, DOFLoc::Vertex, 1},
            {2, DOFLoc::Vertex, 2},
            {3, DOFLoc::Edge, 0},
            {4, DOFLoc::Edge, 1},
            {5, DOFLoc::Edge, 2}}) {
  type = FEType::P2;
}

int P2_FE::get_ndofs() const {
  return 6;
}

int P2_FE::get_order() const {
  return 2;
}

std::vector<double> P2_FE::evaluate_bfs(double xi, double eta) const {
  std::vector<double> bfs(6);
  bfs[0] = 1.0 - 3.0 * xi - 3.0 * eta + 2.0 * xi * xi + 2.0 * eta * eta +
           4.0 * xi * eta;
  bfs[1] = -xi + 2.0 * xi * xi;
  bfs[2] = -eta + 2.0 * eta * eta;
  bfs[3] = 4.0 * xi - 4.0 * xi * xi - 4.0 * xi * eta;
  bfs[4] = 4.0 * xi * eta;
  bfs[5] = 4.0 * eta - 4.0 * eta * eta - 4.0 * xi * eta;
  return bfs;
}

std::vector<std::vector<double>> P2_FE::evaluate_grad_bfs(double xi,
                                                          double eta) const {
  std::vector<std::vector<double>> grad_bfs(6, std::vector<double>(2));
  grad_bfs[0][0] = 4.0 * xi + 4.0 * eta - 3.0;
  grad_bfs[0][1] = 4.0 * xi + 4.0 * eta - 3.0;
  grad_bfs[1][0] = 4.0 * xi - 1.0;
  grad_bfs[1][1] = 0.0;
  grad_bfs[2][0] = 0.0;
  grad_bfs[2][1] = 4.0 * eta - 1.0;
  grad_bfs[3][0] = -8.0 * xi - 4.0 * eta + 4.0;
  grad_bfs[3][1] = -4.0 * xi;
  grad_bfs[4][0] = 4.0 * eta;
  grad_bfs[4][1] = 4.0 * xi;
  grad_bfs[5][0] = -4.0 * eta;
  grad_bfs[5][1] = -4.0 * xi - 8.0 * eta + 4.0;
  return grad_bfs;
}

// basis functions evaluated at quadrature points
// returns quad_val[i][j] = phi_i(x_j)
std::vector<std::vector<double>> P2_FE::bfs_at_quad(
    const std::vector<std::vector<double>>& quad_nodes) const {
  const int n_dofs = get_ndofs();
  const std::size_t n_quads = quad_nodes.size();

  // matrix with dimensions n_dofs x n_quads
  std::vector<std::vector<double>> vals(n_dofs, std::vector<double>(n_quads));

  for (std::size_t i = 0; i < n_dofs; i++) {
    for (std::size_t j = 0; j < n_quads; j++) {
      double xi = quad_nodes.at(j).at(0);
      double eta = quad_nodes.at(j).at(1);

      std::vector<double> bfs = evaluate_bfs(xi, eta);

      vals.at(i).at(j) = bfs.at(i);
    }
  }

  return vals;
}

// gradient of the basis functions evaluated at quadrature points
// returns quad_val[i][j] = grad phi_i(x_j)
std::vector<std::vector<double>> P2_FE::grad_bfs_at_quad(
    const std::vector<std::vector<double>>& quad_nodes) const {
  const int n_dofs = get_ndofs();
  const int n_quads = quad_nodes.size();

  // matrix with dimensions n_dofs x n_quads
  std::vector<std::vector<double>> vals(n_dofs, std::vector<double>(n_quads));

  for (std::size_t i = 0; i < n_dofs; i++) {
    for (std::size_t j = 0; j < n_quads; j++) {
      double xi = quad_nodes.at(j).at(0);
      double eta = quad_nodes.at(j).at(1);

      std::vector<std::vector<double>> grad_bfs = evaluate_grad_bfs(xi, eta);

      vals.at(i).at(j) = grad_bfs.at(i).at(j);
    }
  }

  return vals;
}

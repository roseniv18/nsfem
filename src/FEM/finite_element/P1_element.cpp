#include "P1_element.h"

P1_FE::P1_FE()
    : dofs({{0, DOFLoc::Vertex, 0},
            {1, DOFLoc::Vertex, 1},
            {2, DOFLoc::Vertex, 2}}) {
  type = FEType::P1;
  ref_nodes = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
}

std::vector<DOF> P1_FE::get_dofs() const {
  return dofs;
}

int P1_FE::get_ndofs() const {
  return dofs.size();
}

int P1_FE::get_order() const {
  return 1;
}

std::vector<RefNode> P1_FE::get_ref_nodes() const {
  return ref_nodes;
}

std::vector<double> P1_FE::evaluate_bfs(const RefNode& pt) const {
  std::vector<double> bfs(3);
  const double xi = pt.xi;
  const double eta = pt.eta;

  bfs[0] = 1.0 - xi - eta;
  bfs[1] = xi;
  bfs[2] = eta;
  return bfs;
}

std::vector<std::vector<double>> P1_FE::evaluate_grad_bfs(
    const RefNode& pt) const {
  std::vector<std::vector<double>> grad_bfs(3, std::vector<double>(2));

  grad_bfs[0][0] = -1.0;
  grad_bfs[0][1] = -1.0;
  grad_bfs[1][0] = 1.0;
  grad_bfs[1][1] = 0.0;
  grad_bfs[2][0] = 0.0;
  grad_bfs[2][1] = 1.0;
  return grad_bfs;
}

// basis functions evaluated at quadrature points
// returns quad_val[i][j] = phi_i(x_j)
std::vector<std::vector<double>> P1_FE::bfs_at_quad(
    const std::vector<RefNode>& quad_nodes) const {
  const int n_dofs = get_ndofs();
  const std::size_t n_quads = quad_nodes.size();

  // matrix with dimensions n_dofs x n_quads
  std::vector<std::vector<double>> vals(n_dofs, std::vector<double>(n_quads));

  for (std::size_t i = 0; i < n_dofs; i++) {
    for (std::size_t j = 0; j < n_quads; j++) {
      const RefNode& pt = quad_nodes.at(j);

      std::vector<double> bfs = evaluate_bfs(pt);

      vals.at(i).at(j) = bfs.at(i);
    }
  }

  return vals;
}

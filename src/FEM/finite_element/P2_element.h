#ifndef P2_ELEMENT_H
#define P2_ELEMENT_H

#include <vector>
#include "FEM/finite_element/dof_handler.h"
#include "FEM/quadrature/quadrature.h"
#include "finite_element.h"

// Quadratic Lagrange Triangle
class P2_FE : public FE {
 public:
  P2_FE();

  int get_ndofs() const override;
  int get_order() const override;

  std::vector<double> evaluate_bfs(double xi, double eta) const override;
  std::vector<std::vector<double>> evaluate_grad_bfs(double xi,
                                                     double eta) const override;
  std::vector<std::vector<double>> bfs_at_quad(
      const std::vector<std::vector<double>>& quad_nodes) const override;
  std::vector<std::vector<double>> grad_bfs_at_quad(
      const std::vector<std::vector<double>>& quad_nodes) const override;

 private:
  std::vector<DOF> dofs;
};

#endif

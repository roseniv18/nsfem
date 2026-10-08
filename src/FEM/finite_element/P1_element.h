#ifndef P1_ELEMENT_H
#define P1_ELEMENT_H

#include <vector>
#include "FEM/finite_element/dof.h"
#include "FEM/quadrature/quadrature.h"
#include "finite_element.h"

// Linear Lagrange Triangle
class P1_FE : public FE {
 public:
  P1_FE();

  int get_ndofs() const override;
  int get_order() const override;

  std::vector<DOF> get_dofs() const override;

  std::vector<RefNode> get_ref_nodes() const override;

  std::vector<double> evaluate_bfs(double xi, double eta) const override;
  std::vector<std::vector<double>> evaluate_grad_bfs(double xi,
                                                     double eta) const override;
  std::vector<std::vector<double>> bfs_at_quad(
      const std::vector<std::vector<double>>& quad_nodes) const override;

 private:
  std::vector<DOF> dofs;
  std::vector<RefNode> ref_nodes;
};

#endif

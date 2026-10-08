#ifndef FE_MAPPING
#define FE_MAPPING

#include <Eigen/Dense>
#include "FEM/finite_element/P1_element.h"
#include "FEM/mapping/phys_node.h"
#include "FEM/mapping/ref_node.h"
#include "FEM/quadrature/quadrature.h"
#include "mesh/mesh.h"

using Eigen::MatrixXd, Eigen::VectorXd;

class FEMap2D {
 public:
  FEMap2D(const Element& element,
          const std::vector<Node>& nodes,
          const FE& finite_element);

  //   access functions
  MatrixXd jacobian() const;
  MatrixXd jacobianInvT() const;
  double det_jacobian() const;
  std::vector<PhysNode> get_phys_grads() const;
  std::vector<PhysNode> get_phys_coords() const;

 private:
  MatrixXd J{2, 2};
  MatrixXd JinvT{2, 2};
  double detJ{};
  std::vector<PhysNode> phys_grads;
  std::vector<PhysNode> phys_coords;
};

#endif

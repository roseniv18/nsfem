#ifndef FE_MAPPING
#define FE_MAPPING

#include <Eigen/Dense>
#include "FEM/finite_element/P1_element.h"
#include "FEM/quadrature/quadrature.h"
#include "mesh/mesh.h"
#include "point2d.h"

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
  std::vector<Point2D> get_phys_grads() const;
  std::vector<Point2D> get_phys_coords() const;

 private:
  MatrixXd J{2, 2};
  MatrixXd JinvT{2, 2};
  double detJ{};
  std::vector<Point2D> phys_grads;
  std::vector<Point2D> phys_coords;
};

#endif

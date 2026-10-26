#ifndef FE_MAPPING
#define FE_MAPPING

#include <Eigen/Dense>
#include <vector>

#include "FEM/finite_element/finite_element.h"
#include "FEM/mapping/phys_node.h"
#include "FEM/mapping/ref_node.h"
#include "mesh/mesh.h"

using Eigen::Matrix2d, Eigen::Vector2d;

namespace nsfem::mapping {
using nsfem::fem::FE;
using nsfem::mesh::Element, nsfem::mesh::Node;

class FEMap2D {
 public:
  FEMap2D(const Element& element,
          const std::vector<Node>& nodes,
          const FE& geometry_fe);

  //   access functions
  PhysNode physical_point(const RefNode& pt) const;

  Matrix2d jacobian(const RefNode& pt) const;

  Matrix2d jacobianInvT(const RefNode& pt) const;

  double det_jacobian(const RefNode& pt) const;

 private:
  const std::vector<Node>& nodes;
  const FE& element;
};
}  // namespace nsfem::mapping

#endif

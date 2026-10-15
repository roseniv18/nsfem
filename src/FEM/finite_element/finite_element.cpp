#include "finite_element.h"
#include "P1_element.h"
#include "P2_element.h"
#include "mesh/mesh.h"

std::unique_ptr<FE> FE::build_fe_type(const FEType type) {
  switch (type) {
    case FEType::P1:
      return std::make_unique<P1_FE>();
    case FEType::P2:
      return std::make_unique<P2_FE>();
    default:
      throw std::invalid_argument("Unknown FE Type");
  }
}

std::unique_ptr<FE> FE::build_geo_type(const ElementType& type) {
  switch (type) {
    case ElementType::Triangle3:
      return FE::build_fe_type(FEType::P1);

    case ElementType::Triangle6:
      return FE::build_fe_type(FEType::P2);

    default:
      throw std::invalid_argument(
          "Unsupported element type for 2D geometry mapping!");
  }
}
